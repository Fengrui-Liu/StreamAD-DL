import os
from datetime import datetime
from time import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from hydra import compose, initialize
from hydra.utils import instantiate
from tqdm import tqdm

from src.dlutils import EarlyStopping
from src.eval import eval
from src.train import train
from src.utils import color

initialize(version_base="1.1", config_path="./", job_name="train")
cfg = compose(config_name="train", overrides=[])

now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tb_writer_train = SummaryWriter(
    log_dir=cfg.general.output_dir
    + "/"
    + str(cfg.model.model_name)
    + "/"
    + str(cfg.model.data_name)
    + "/"
    + str(now),
    filename_suffix="-train",
)
tb_writer_eval = SummaryWriter(
    log_dir=cfg.general.output_dir
    + "/"
    + str(cfg.model.model_name)
    + "/"
    + str(cfg.model.data_name)
    + "/"
    + str(now),
    filename_suffix="-test",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, optimizer, scheduler):
    folder = f"./checkpoints/{model.name}_{cfg.data.data_name}/"
    os.makedirs(folder, exist_ok=True)
    file_path = f"{folder}/model.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        file_path,
    )


def run_train(model, train_iter, test_iter, optimizer, criterion, scheduler):
    print(
        f"{color.HEADER}Training {cfg.model.model_name} on {cfg.data.data_name}{color.ENDC}"
    )

    if not os.path.exists(cfg.general.output_dir):
        os.makedirs(cfg.general.output_dir)

    num_epochs = cfg.training.num_epochs
    global_step = 0
    train_loss = 0.0
    logging_loss = 0.0

    early_stopping = EarlyStopping()

    for e in tqdm(
        range(0, num_epochs),
        unit=" Epoch",
        desc="Training Epoch",
    ):
        for i, batch in tqdm(
            enumerate(train_iter),
            total=len(train_iter),
            unit=" Batch",
            desc="Training Batch",
        ):

            optimizer.zero_grad()

            loss = train(model=model, criterion=criterion, batch=batch)
            train_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % cfg.general.logging_steps == 0:

                tb_writer_train.add_scalar(
                    "train_loss",
                    (train_loss - logging_loss) / cfg.general.logging_steps,
                    global_step,
                )
                tb_writer_train.add_scalar(
                    "learning rate", scheduler.get_last_lr()[0], global_step
                )

                logging_loss = train_loss

        if cfg.general.eval_during_training:
            eval_loss = run_eval(
                model=model,
                test_iter=test_iter,
                criterion=criterion,
                global_step=global_step,
            )

            if early_stopping(eval_loss):
                save_model(model, optimizer, scheduler)
                print(
                    f"{color.HEADER}Early Stop {cfg.model.model_name} on {cfg.data.data_name} at Epoch {e} {color.ENDC}"
                )
                return

    save_model(model, optimizer, scheduler)
    return


def run_eval(model, test_iter, criterion, global_step):

    eval_loss = 0.0

    for i, batch in tqdm(
        enumerate(test_iter), total=len(test_iter), desc="Evaluating"
    ):
        with torch.no_grad():
            loss, prediction, target = eval(
                model=model, criterion=criterion, batch=batch
            )
            loss = loss.item()

            eval_loss += loss

    tb_writer_eval.add_scalar(
        "eval_loss",
        eval_loss / len(test_iter),
        global_step,
    )

    return eval_loss / len(test_iter)


def run_predict(model, X, criterion):
    with torch.no_grad():
        loss, x_hat, target = eval(
            model=model, criterion=criterion, batch=(X, X)
        )
        loss = loss.item()

    return loss, x_hat, target


def run():

    # Load data
    ts = instantiate(cfg.data)
    train_iter, test_iter, nb_features = ts.get_loaders()

    # Set model dim as nb_features, consistnet with the inputs.
    cfg.model.dim = nb_features

    # Load model
    md = instantiate(cfg.model)
    model, optimizer, scheduler = md.get_model()

    criterion = nn.MSELoss()

    # Training phase
    if cfg.general.do_train:

        start = time()

        run_train(
            model=model,
            train_iter=train_iter,
            test_iter=test_iter,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
        )

        print(
            f"{color.BOLD}Training time: {round(time() - start,4)} s{color.ENDC}"
        )

    if cfg.general.do_predict:
        import random

        X = torch.DoubleTensor(
            [[[random.random()] for i in range(cfg.data.seq_length)]]
        ).to(device)
        for i in range(10, 15):
            X[0][i][0] *= 10
        print(X.shape)
        loss, x_hat, target = run_predict(model=model, X=X, criterion=criterion)
        print(loss, "\n", x_hat.view(-1), "\n", target.view(-1))


if __name__ == "__main__":
    run()
