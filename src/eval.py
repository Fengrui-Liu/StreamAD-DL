import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dagmm_eval(model, criterion, batch):

    x, target = batch
    _, x_hat, _, _ = model(x.to(device))
    loss = criterion(x_hat, target)

    return loss, x_hat, target


def attention_eval(model, criterion, batch):

    x, target = batch
    x = x.permute(1, 0, 2)
    x_hat = model(x.to(device))
    x_hat = x_hat.permute(1, 0, 2)
    loss = criterion(x_hat, target)

    return loss, x_hat, target


def omnianomaly_eval(model, criterion, batch):

    x, target = batch
    x_hat, _, _ = model(x.to(device))
    loss = criterion(x_hat, target)

    return loss, x_hat, target


def usad_eval(model, criterion, batch):

    x, target = batch
    x_hat_1, x_hat_2, x_hat_2_1 = model(x.to(device))
    loss = 0.1 * criterion(x_hat_1, target) + 0.9 * criterion(x_hat_2_1, target)
    x_hat = 0.1 * x_hat_1 + 0.9 * x_hat_2_1

    return loss, x_hat, target


def tranad_eval(model, criterion, batch):

    x, target = batch
    x = x.permute(1, 0, 2)
    _, x_hat = model(x.to(device))
    x_hat = x_hat.permute(1, 0, 2)
    loss = criterion(x_hat, target)

    return loss, x_hat, target


def eval(model, criterion, batch):
    model.eval()

    model_name = str.lower(model.name)

    eval_models = {
        "dagmm": dagmm_eval,
        "attention": attention_eval,
        "omnianomaly": omnianomaly_eval,
        "usad": usad_eval,
        "tranad": tranad_eval,
    }

    assert model_name in eval_models, f"Model {model.name} not implemented"

    loss, x_hat, target = eval_models[model_name](
        model=model,
        criterion=criterion,
        batch=batch,
    )

    return loss, x_hat, target
