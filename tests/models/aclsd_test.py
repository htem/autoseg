import unittest
import torch
from autoseg.networks import setup_unet
from autoseg.utils import neighborhood
from autoseg.models import ACLSDModel


class TestACLSDModel(unittest.TestCase):

    def setUp(self):
        # Set up any required data or configuration for your tests
        unet = setup_unet(
            in_channels=10,
            downsample_factors=[(2, 2, 2), (2, 2, 2)],
            padding="same",
            num_heads=1,
        )        
        self.mtlsd_model = ACLSDModel(unet, unet.out_channels)

    def test_forward(self):
        # Test the forward method of STELARRModel
        input_tensor = torch.randn((1, 10, 100, 100, 100))
        affs = self.mtlsd_model(input_tensor)

        # Check if the output tensors have the correct shapes
        self.assertEqual(affs.shape, (1, len(neighborhood), 100, 100, 100))

        # Check if the values are within a reasonable range (this can be adjusted based on your model)
        self.assertTrue(torch.all(affs >= 0) and torch.all(affs <= 1))
