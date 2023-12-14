import torch

from ..networks.UNet import UNet, ConvPass
from ..utils import neighborhood


class ACLSDModel(torch.nn.Module):
    def __init__(self, unet: UNet, num_fmaps: int):
        """
        ACLSD (Auto-Context LSD) Segmentation Model

        Initializes an ACLSDModel instance.

        Parameters:
            unet (UNet): The U-Net architecture used in the ACLSD model.
            num_fmaps (int): The number of feature maps for the convolutional pass.

        Attributes:
            unet (UNet): The U-Net architecture used in the ACLSD model.
            aff_head (ConvPass): Convolutional pass for affinity predictions.
        """
        super(ACLSDModel, self).__init__()

        self.unet: UNet = unet
        self.aff_head: ConvPass = ConvPass(
            num_fmaps, len(neighborhood), [[1, 1, 1]], activation="Sigmoid"
        )

    def forward(self, input):
        """
        Forward pass of the ACLSDModel.

        Performs a forward pass through the U-Net and the convolutional pass for affinity predictions.

        Parameters:
            input: Input data for the model.

        Returns:
            affs: Affinity predictions produced by the model.
        """
        x = self.unet(input)
        affs = self.aff_head(x)
        return affs
