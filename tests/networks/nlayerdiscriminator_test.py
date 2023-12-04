import unittest
import torch
from autoseg.networks.NLayerDiscriminator import *


class TestNLayerDiscriminator2D(unittest.TestCase):
    def test_output_shape(self):
        # Test that the discriminator produces an output tensor of the expected shape
        input_nc = 3
        ngf = 64
        n_layers = 3
        netD = NLayerDiscriminator2D(input_nc, ngf, n_layers)

        # Create a random input tensor
        input_tensor = torch.randn((1, input_nc, 256, 256))

        # Pass the input tensor through the discriminator
        output_tensor = netD(input_tensor)

        # Check that the output tensor has the expected shape
        expected_shape = (1, 1, 30, 30)
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_fov(self):
        # Test that the field of view (FOV) calculation is correct
        input_nc = 3
        ngf = 64
        n_layers = 3
        netD = NLayerDiscriminator2D(input_nc, ngf, n_layers)

        # Calculate the FOV of the discriminator
        fov = netD.FOV

        # Check that the FOV is within the expected range (example range)
        expected_fov = 70
        self.assertGreaterEqual(fov, expected_fov - 40)
        self.assertLessEqual(fov, expected_fov + 40)


class TestNLayerDiscriminator3D(unittest.TestCase):
    def test_output_shape(self):
        # Test the output shape of the discriminator
        batch_size = 2
        input_nc = 1
        input_size = (32, 32, 32)
        x = torch.randn(batch_size, input_nc, *input_size)
        net = NLayerDiscriminator3D(input_nc=input_nc)
        output = net(x)
        self.assertEqual(output.shape, (batch_size, 1, 2, 2, 2))

    def test_forward(self):
        # Test the forward pass of the discriminator
        batch_size = 2
        input_nc = 1
        input_size = (32, 32, 32)
        x = torch.randn(batch_size, input_nc, *input_size)
        net = NLayerDiscriminator3D(input_nc=input_nc)
        output = net(x)
        self.assertIsInstance(output, torch.Tensor)

    def test_parameter_count(self):
        # Test the number of parameters in the discriminator
        net = NLayerDiscriminator3D()
        num_params = sum(p.numel() for p in net.parameters())
        self.assertEqual(num_params, 11048769)


class TestNLayerDiscriminator(unittest.TestCase):
    def test_init_2d(self):
        discriminator = NLayerDiscriminator(ndims=2, input_nc=3, ngf=64, n_layers=3)
        self.assertIsInstance(discriminator, NLayerDiscriminator)
        self.assertIsInstance(discriminator.model, torch.nn.Sequential)
        self.assertEqual(len(discriminator.model), 12)

    def test_init_3d(self):
        discriminator = NLayerDiscriminator(ndims=3, input_nc=1, ngf=32, n_layers=5)
        self.assertIsInstance(discriminator, NLayerDiscriminator)
        self.assertIsInstance(discriminator.model, torch.nn.Sequential)
        self.assertEqual(len(discriminator.model), 18)

    def test_init_invalid_ndims(self):
        with self.assertRaises(ValueError):
            discriminator = NLayerDiscriminator(ndims=4, input_nc=1, ngf=32, n_layers=5)
