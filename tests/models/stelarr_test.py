import unittest
import torch
from autoseg.networks import setup_unet
from autoseg.utils import neighborhood
from autoseg.models import STELARRModel


class TestSTELARRModel(unittest.TestCase):
    def setUp(self):
        # Set up any required data or configuration for your tests
        unet = setup_unet(downsample_factors=[(2, 2, 2), (2, 2, 2)], num_heads=3)
        self.stelarr_model = STELARRModel(unet, unet.out_channels)

    def test_forward(self):
        # Test the forward method of STELARRModel
        input_tensor = torch.randn((1, 1, 100, 100, 100))
        lsds, affs, fake = self.stelarr_model(input_tensor)

        # Check if the output tensors have the correct shapes
        self.assertEqual(lsds.shape, (1, 10, 60, 60, 60))
        self.assertEqual(affs.shape, (1, len(neighborhood), 60, 60, 60))
        self.assertEqual(fake.shape, (1, 1, 60, 60, 60))

        # Check if the values are within a reasonable range (this can be adjusted based on your model)
        self.assertTrue(torch.all(lsds >= 0) and torch.all(lsds <= 1))
        self.assertTrue(torch.all(affs >= 0) and torch.all(affs <= 1))

        self.assertTrue(torch.all(fake >= 0) and torch.all(fake <= 1))
