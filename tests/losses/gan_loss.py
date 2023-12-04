import torch
import unittest
from autoseg.losses import GANLoss


class TestGANLoss(unittest.TestCase):

    def setUp(self):
        # Initialize GANLoss with default settings
        self.gan_loss = GANLoss()

    def test_lsgan_loss_real(self):
        # Test LSGAN loss for real prediction
        real_pred = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        loss = self.gan_loss(real_pred=real_pred)
        expected_loss = torch.nn.MSELoss()(real_pred, torch.full_like(real_pred, 1.0))
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_lsgan_loss_fake(self):
        # Test LSGAN loss for fake prediction
        fake_pred = torch.tensor([0.2, 0.3, 0.1], requires_grad=True)
        loss = self.gan_loss(fake_pred=fake_pred)
        expected_loss = torch.nn.MSELoss()(fake_pred, torch.full_like(fake_pred, 0.0))
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_vanilla_loss_real(self):
        # Test Vanilla GAN loss for real prediction
        self.gan_loss = GANLoss(gan_mode="vanilla")
        real_pred = torch.tensor([0.9, 0.8, 0.7], requires_grad=True)
        loss = self.gan_loss(real_pred=real_pred)
        expected_loss = torch.nn.BCEWithLogitsLoss()(real_pred, torch.full_like(real_pred, 1.0))
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_vanilla_loss_fake(self):
        # Test Vanilla GAN loss for fake prediction
        self.gan_loss = GANLoss(gan_mode="vanilla")
        fake_pred = torch.tensor([0.2, 0.3, 0.1], requires_grad=True)
        loss = self.gan_loss(fake_pred=fake_pred)
        expected_loss = torch.nn.BCEWithLogitsLoss()(fake_pred, torch.full_like(fake_pred, 0.0))
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_wgangp_loss_real(self):
        # Test WGAN-GP loss for real prediction
        self.gan_loss = GANLoss(gan_mode="wgangp")
        real_pred = torch.tensor([-0.9, -0.8, -0.7], requires_grad=True)
        loss = self.gan_loss(real_pred=real_pred)
        expected_loss = -real_pred.mean()
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_wgangp_loss_fake(self):
        # Test WGAN-GP loss for fake prediction
        self.gan_loss = GANLoss(gan_mode="wgangp")
        fake_pred = torch.tensor([0.2, 0.3, 0.1], requires_grad=True)
        loss = self.gan_loss(fake_pred=fake_pred)
        expected_loss = fake_pred.mean()
        self.assertTrue(torch.allclose(loss, expected_loss))
