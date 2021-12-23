from torch.optim import SGD


def setup_optimiser(model, learning_rate, momentum, weight_decay):
    return SGD(
        model.parameters(),
        learning_rate,
        momentum,
        weight_decay
    )
