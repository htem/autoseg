import unittest
import torch
from autoseg.networks import setup_unet
from autoseg.utils import neighborhood
from autoseg.models import MTLSDModel


class TestMTLSDModel(unittest.TestCase):

    def setUp(self):
        # Set up any required data or configuration for your tests
        unet = setup_unet()
        self.mtlsd_model = MTLSDModel(unet, unet.out_channels)

    def test_forward(self):
        # Test the forward method of STELARRModel
        input_tensor = torch.randn((1, 1, 100, 100, 100))
        lsds, affs = self.mtlsd_model(input_tensor)

        # Check if the output tensors have the correct shapes
        self.assertEqual(lsds.shape, (1, 10, 8, 8, 8))
        self.assertEqual(affs.shape, (1, len(neighborhood), 8, 8, 8))

        # Check if the values are within a reasonable range (this can be adjusted based on your model)
        self.assertTrue(torch.all(lsds >= 0) and torch.all(lsds <= 1))
        self.assertTrue(torch.all(affs >= 0) and torch.all(affs <= 1))
