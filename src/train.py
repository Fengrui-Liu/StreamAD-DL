from src.dlutils import ComputeLoss
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dagmm_train(model, criterion, batch):

    x, target = batch
    _, x_hat, z, gamma = model(x.to(device))
    loss1 = criterion(x_hat, target)
    loss2 = criterion(gamma, target)
    loss = torch.mean(loss1) + torch.mean(loss2)

    return loss


def attention_train(model, criterion, batch):

    x, target = batch
    x_hat = model(x.to(device))
    loss = criterion(x_hat, target)
    loss = torch.mean(loss)

    return loss


def omnianomaly_train(model, criterion, batch):

    x, target = batch
    x_hat, mu, logvar = model(x.to(device))
    mse_loss = criterion(x_hat, target)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse_loss + 0.1 * kld

    return loss


def usad_train(model, criterion, batch):

    x, target = batch
    x_hat_1, x_hat_2, x_hat_2_1 = model(x.to(device))

    n = model.forward_n // 1000 + 1
    loss1 = criterion(x_hat_1, target) / n + (1 - 1 / n) * criterion(
        x_hat_2_1, target
    )

    loss2 = criterion(x_hat_2, target) / n - (1 - 1 / n) * criterion(
        x_hat_2_1, target
    )

    loss1.backward(retain_graph=True)

    return loss2


def tranad_train(model, criterion, batch):

    x, target = batch

    target = target.permute(1, 0, 2)
    x = x.permute(1, 0, 2)
    x_hat_1, x_hat_2 = model(x.to(device))

    n = model.forward_n // 1000 + 1

    loss = criterion(x_hat_1, target) / n + (1 - 1 / n) * criterion(
        x_hat_2, target
    )

    return loss


def train(model, criterion, batch):

    model.train()
    model_name = str.lower(model.name)

    train_models = {
        "dagmm": dagmm_train,
        "attention": attention_train,
        "omnianomaly": omnianomaly_train,
        "usad": usad_train,
        "tranad": tranad_train,
    }

    assert model_name in train_models, f"Model {model.name} not implemented"

    loss = train_models[model_name](
        model=model, criterion=criterion, batch=batch
    )
    loss.backward()

    return loss
