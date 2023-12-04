import torch
import unittest
from autoseg.losses import Weighted_MSELoss


class DummyDiscriminator(torch.nn.Module):
    def forward(self, x):
        return torch.rand_like(x)

class TestWeightedMSELoss(unittest.TestCase):

    def setUp(self):
        discrim = DummyDiscriminator()
        self.weighted_mseloss = Weighted_MSELoss(discrim=discrim)

    def test_calc_loss(self):
        prediction = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
        weights = torch.tensor([1.0, 0.0, 1.0], requires_grad=False)

        loss = self.weighted_mseloss._calc_loss(prediction, target, weights)

        # Add your assertion here based on the expected output
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)

    def test_forward(self):
        # Create dummy input data
        pred_lsds = torch.randn(3, requires_grad=True)
        gt_lsds = torch.randn(3, requires_grad=True)
        lsds_weights = torch.randn(3, requires_grad=False)
        pred_affs = torch.randn(3, requires_grad=True)
        gt_affs = torch.randn(3, requires_grad=True)
        affs_weights = torch.randn(3, requires_grad=False)
        pred_enhanced = torch.randn(3, requires_grad=True)
        gt_enhanced = torch.randn(3, requires_grad=True)

        # Call the forward method
        loss = self.weighted_mseloss(
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
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)