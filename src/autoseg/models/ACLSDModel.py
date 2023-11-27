import torch

from ..networks.UNet import UNet, ConvPass
from ..utils import neighborhood


class ACLSDModel(torch.nn.Module):
    def __init__(self, unet: UNet, num_fmaps: int):
        """
           ACLSD (Auto-Context LSD) Segmentation Model
        """
        super(ACLSDModel, self).__init__()

        self.unet: UNet = unet
        self.aff_head: ConvPass = ConvPass(num_fmaps, len(neighborhood), [[1, 1, 1]], activation="Sigmoid")

    def forward(self, input):
        x = self.unet(input)
        affs = self.aff_head(x)
        return affs