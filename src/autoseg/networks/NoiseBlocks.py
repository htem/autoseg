import torch


class NoiseBlock(torch.nn.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean=0 and stdev=1"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = 1  # only make one noise feature
        noise = torch.empty(shape, device=x.device).normal_()
        return torch.cat([x, noise.requires_grad_()], 1)


class ParameterizedNoiseBlock(torch.nn.Module):
    """Definies a block for producing and appending a feature map of gaussian noise with mean and stdev defined by the first two feature maps of the incoming tensor"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        noise = torch.normal(x[:, 0, ...], torch.relu(x[:, 1, ...])).unsqueeze(1)
        return torch.cat([x, noise.requires_grad_()], 1)