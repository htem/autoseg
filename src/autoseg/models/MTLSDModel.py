import torch

from ..networks.UNet import UNet, ConvPass
from ..utils import neighborhood


class MTLSDModel(torch.nn.Module):
    def __init__(self, unet: UNet, num_fmaps: int):
        super(MTLSDModel, self).__init__()

        self.unet: UNet = unet
        self.lsd_head: ConvPass = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid")
        self.aff_head: ConvPass = ConvPass(num_fmaps, len(neighborhood), [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        lsds = self.lsd_head(x[0])
        affs = self.aff_head(x[1])
        return lsds, affs