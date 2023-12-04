import torch
import unittest
from autoseg.losses import Weighted_MSELoss


class DummyDiscriminator(torch.nn.Module):
    def forward(self, x):
        return torch.rand_like(x)
    
class TestWeightedMSELoss(unittest.TestCase):

    def setUp(self):
        # Initialize Weighted_MSELoss with default settings
        discrim = DummyDiscriminator()
        self.weighted_mse_loss = Weighted_MSELoss(discrim=discrim)

    def test_calc_loss_with_weights(self):
        # Test _calc_loss method with weights provided
        prediction = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
        weights = torch.tensor([1.0, 0.0, 1.0], requires_grad=False)

        loss = self.weighted_mse_loss._calc_loss(prediction, target, weights)

        expected_loss = weights * (prediction - target) ** 2
        if len(torch.nonzero(expected_loss)) != 0 and type(weights)==torch.Tensor:
            mask = torch.masked_select(expected_loss, torch.gt(weights, 0))
            expected_loss = torch.mean(mask)
        else:
            expected_loss = torch.mean(expected_loss)

        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_calc_loss_without_weights(self):
        # Test _calc_loss method without weights provided
        prediction = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)

        loss = self.weighted_mse_loss._calc_loss(prediction, target)

        expected_loss = torch.mean((prediction - target) ** 2)
        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_forward_with_gan_loss(self):
        # Test forward method with GAN loss component
        pred_lsds = torch.randn(3, requires_grad=True)
        gt_lsds = torch.randn(3, requires_grad=True)
        lsds_weights = torch.randn(3, requires_grad=False)
        pred_affs = torch.randn(3, requires_grad=True)
        gt_affs = torch.randn(3, requires_grad=True)
        affs_weights = torch.randn(3, requires_grad=False)
        pred_enhanced = torch.randn(3, requires_grad=True)
        gt_enhanced = torch.randn(3, requires_grad=True)

        loss = self.weighted_mse_loss(
            pred_lsds=pred_lsds,
            gt_lsds=gt_lsds,
            lsds_weights=lsds_weights,
            pred_affs=pred_affs,
            gt_affs=gt_affs,
            affs_weights=affs_weights,
            pred_enhanced=pred_enhanced,
            gt_enhanced=gt_enhanced,
        )

        # Add your assertion here based on the expected output
        # For example, assert that the loss is a tensor and requires gradient
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)

# Add more tests as needed

if __name__ == '__main__':
    unittest.main()
