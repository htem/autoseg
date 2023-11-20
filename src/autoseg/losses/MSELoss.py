import torch


class Weighted_MSELoss(torch.nn.Module):
    def __init__(self, aff_lambda=1.0, gan_lambda=1.0, discrim=None):
        super(Weighted_MSELoss, self).__init__()
        self.aff_lambda = aff_lambda
        self.gan_lambda = gan_lambda
        self.mse_loss = torch.nn.MSELoss()
        self.discriminator = discrim
    
    def _calc_loss(self, prediction, target, weights=None):
        if weights is not None:
            scaled = (prediction - target) ** 2
        else:
            scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        pred_lsds=None,
        gt_lsds=None,
        lsds_weights=None,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
        pred_enhanced=None,
        gt_enhanced=None,
    ):
        # Calculate MSE loss for LSD and Affs
        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        aff_loss = self.aff_lambda * self._calc_loss(pred_affs, gt_affs, affs_weights)

        # calculate MSE loss for GAN errors, pass data through discrim network in process
        if gt_enhanced is not None and pred_enhanced is not None:
            real_scores = self.discriminator(gt_enhanced)
            fake_scores = self.discriminator(pred_enhanced)
            gan_loss = self.gan_lambda * (torch.mean((real_scores - 1) ** 2) + torch.mean(fake_scores ** 2))
        else:
            gan_loss: float = 0.

        return lsd_loss + aff_loss + gan_loss