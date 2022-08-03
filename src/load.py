import os
from typing import List

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from joblib import dump
import src.models
from src.utils import color

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoadTimeSeriesDataset(object):
    def __init__(
        self,
        data_path: str,
        data_name: str,
        categorical_cols: List[str],
        use_cols: List[str],
        index_col: str,
        seq_length: int,
        batch_size: int,
        train_size: float,
    ):
        """
        :param data_path: path to datafile
        :param categorical_cols: name of the categorical columns, if None pass empty list
        :param index_col: column to use as index
        :param seq_length: window length to use
        :param batch_size:
        """

        self.data_name = data_name
        self.data = pd.read_csv(
            data_path, index_col=index_col, usecols=use_cols
        )
        self.categorical_cols = categorical_cols
        self.numerical_cols = list(
            set(self.data.columns) - set(categorical_cols)
        )

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.train_size = train_size

        transformations = [
            (
                "scaler",
                RobustScaler(
                    with_centering=False,
                    quantile_range=(1, 99),
                ),
                self.numerical_cols,
            )
        ]
        if len(self.categorical_cols) > 0:
            transformations.append(
                ("encoder", OneHotEncoder(), self.categorical_cols)
            )
        self.preprocessor = ColumnTransformer(
            transformations, remainder="passthrough"
        )

    def preprocess_data(self):
        """Preprocessing function"""

        X_train, X_test = train_test_split(
            self.data, train_size=self.train_size, shuffle=False
        )
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        folder = f"./data_scaler/{self.data_name}/"
        os.makedirs(folder, exist_ok=True)
        fname = f"{folder}/scaler.joblib"
        dump(self.preprocessor, fname)
        return X_train, X_test

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """

        nb_obs, nb_features = X.shape
        features = []

        for i in range(0, nb_obs - self.seq_length):
            features.append(
                torch.DoubleTensor(X[i : i + self.seq_length, :]).unsqueeze(0)
            )

        features_var = torch.cat(features)

        return TensorDataset(features_var, features_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset

        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train)
        test_dataset = self.frame_series(X_test)

        train_iter = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)
            ),
        )
        test_iter = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)
            ),
        )
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """
        Inverts the scale of the predictions
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.numpy()

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        unscaled = self.preprocessor.named_transformers_[
            "scaler"
        ].inverse_transform(predictions)
        return torch.Tensor(unscaled)


class LoadModel(object):
    def __init__(
        self,
        model_name,
        dim,
        data_name,
        lr,
        lrs_step_size,
        seq_len,
        retrain=False,
    ):
        self.model_name = model_name
        self.dim = dim
        self.retrain = retrain
        self.seq_len = seq_len
        self.lr = lr
        self.lrs_step_size = lrs_step_size
        self.data_name = data_name

    def get_model(self):
        model_class = getattr(src.models, self.model_name)
        model = (
            model_class(feats=self.dim, n_window=self.seq_len)
            .double()
            .to(device)
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.lrs_step_size, 0.6
        )
        fname = f"./checkpoints/{self.model_name}_{self.data_name}/model.ckpt"
        if os.path.exists(fname) and (not self.retrain):
            print(
                f"{color.GREEN}Loading pre-trained model: {self.model_name}{color.ENDC}"
            )
            checkpoint = torch.load(fname, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(
                f"{color.GREEN}Creating new model: {self.model_name}{color.ENDC}"
            )

        return (
            model,
            optimizer,
            scheduler,
        )
