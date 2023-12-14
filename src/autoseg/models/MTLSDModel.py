import torch

from ..networks.UNet import UNet, ConvPass
from ..utils import neighborhood


class MTLSDModel(torch.nn.Module):
    def __init__(self, unet: UNet, num_fmaps: int):
        """
        MTLSD (Multi-Task LSD) Segmentation Model

        Initializes an MTLSDModel instance.

        Parameters:
            unet (UNet): The U-Net architecture used in the MTLSD model.
            num_fmaps (int): The number of feature maps for the convolutional passes.

        Attributes:
            unet (UNet): The U-Net architecture used in the MTLSD model.
            lsd_head (ConvPass): Convolutional pass for LSD (Label Set Distribution) predictions.
            aff_head (ConvPass): Convolutional pass for affinity predictions.
        """
        super(MTLSDModel, self).__init__()

        self.unet: UNet = unet
        self.lsd_head: ConvPass = ConvPass(
            num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid"
        )
        self.aff_head: ConvPass = ConvPass(
            num_fmaps, len(neighborhood), [[1, 1, 1]], activation="Sigmoid"
        )

    def forward(self, input):
        """
        Forward pass of the MTLSDModel.

        Performs a forward pass through the U-Net and separate convolutional passes for LSD and affinity predictions.

        Parameters:
            input: Input data for the model.

        Returns:
            lsds: LSD predictions produced by the model.
            affs: Affinity predictions produced by the model.
        """
        x = self.unet(input)
        lsds = self.lsd_head(x[0])
        affs = self.aff_head(x[1])
        return lsds, affs
