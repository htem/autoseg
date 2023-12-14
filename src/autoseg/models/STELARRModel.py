import torch

from ..networks.UNet import UNet, ConvPass
from ..utils import neighborhood


class STELARRModel(torch.nn.Module):
    def __init__(self, unet: UNet, num_fmaps: int) -> None:
        """
        Selective-Task Enhancement, LSD, Affinity, Rine & Repeat (STELARR) Segmentation Model

        Initializes a STELARRModel instance.

        Parameters:
            unet (UNet): The U-Net architecture used in the STELARR model.
            num_fmaps (int): The number of feature maps for the convolutional passes.

        Attributes:
            unet (UNet): The U-Net architecture used in the STELARR model.
            lsd_head (ConvPass): Convolutional pass for LSD (Label Set Distribution) predictions.
            aff_head (ConvPass): Convolutional pass for affinity predictions.
            enhancement_head (ConvPass): Convolutional pass for task-specific enhancement predictions.
        """
        super(STELARRModel, self).__init__()

        self.unet: UNet = unet
        self.lsd_head: ConvPass = ConvPass(
            num_fmaps, 10, [[1, 1, 1]], activation="Sigmoid"
        )
        self.aff_head: ConvPass = ConvPass(
            num_fmaps, len(neighborhood), [[1, 1, 1]], activation="Sigmoid"
        )
        self.enhancement_head: ConvPass = ConvPass(
            num_fmaps, 1, [[1, 1, 1]], activation="Sigmoid"
        )

    def forward(self, input):
        """
        Forward pass of the STELARRModel.

        Performs a forward pass through the U-Net and separate convolutional passes for LSD, affinity, and enhancement predictions.

        Parameters:
            input: Input data for the model.

        Returns:
            lsds: LSD predictions produced by the model.
            affs: Affinity predictions produced by the model.
            fake: Task-specific enhancement predictions produced by the model.
        """
        x = self.unet(input)
        lsds = self.lsd_head(x[0])
        affs = self.aff_head(x[1])
        fake = self.enhancement_head(x[2])

        return lsds, affs, fake
