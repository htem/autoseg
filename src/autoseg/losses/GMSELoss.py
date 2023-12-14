import torch


class Weighted_GMSELoss(torch.nn.Module):
    """
    Weighted GAN Mean Squared Error (GMSE) Loss with GAN Loss.

    This loss combines the traditional MSE loss for LSD and Affinities with an additional
    GAN (Generative Adversarial Network) loss term for enhanced data.

    Parameters:
        aff_lambda (float, optional):
            Weighting factor for the affinity loss. Default is 1.0.
        gan_lambda (float, optional):
            Weighting factor for the GAN loss. Default is 1.0.
        discrim (torch.nn.Module, optional):
            Discriminator network for GAN loss.
    """

    def __init__(self, aff_lambda=1.0, gan_lambda=1.0, discrim=None):
        """
        Initializes the Weighted_MSELoss.

        Args:
            aff_lambda (float, optional):
                Weighting factor for the affinity loss. Default is 1.0.
            gan_lambda (float, optional):
                Weighting factor for the GAN loss. Default is 1.0.
            discrim (torch.nn.Module, optional):
                Discriminator network for GAN loss.
        """
        super(Weighted_GMSELoss, self).__init__()
        self.aff_lambda = aff_lambda
        self.gan_lambda = gan_lambda
        self.mse_loss = torch.nn.MSELoss()
        self.discriminator = discrim

    def _calc_loss(self, prediction, target, weights=None):
        """
        Calculates the weighted mean squared error loss.

        Args:
            prediction (torch.Tensor):
                Predicted values.
            target (torch.Tensor):
                Ground truth values.
            weights (torch.Tensor, optional):
                Weighting factor for each value.

        Returns:
            torch.Tensor:
                Weighted mean squared error loss.
        """
        if type(weights) != torch.Tensor:
            scaled = (prediction - target) ** 2
        else:
            scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0 and type(weights) == torch.Tensor:
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
        """
        Calculates the weighted MSE loss with GAN loss.

        Args:
            pred_lsds (torch.Tensor):
                Predicted LSD values.
            gt_lsds (torch.Tensor):
                Ground truth LSD values.
            lsds_weights (torch.Tensor, optional):
                Weighting factor for each LSD value.
            pred_affs (torch.Tensor):
                Predicted affinity values.
            gt_affs (torch.Tensor):
                Ground truth affinity values.
            affs_weights (torch.Tensor, optional):
                Weighting factor for each affinity value.
            pred_enhanced (torch.Tensor):
                Predicted enhanced data.
            gt_enhanced (torch.Tensor):
                Ground truth enhanced data.

        Returns:
            torch.Tensor:
                Combined weighted MSE loss with GAN loss.
        """
        # calculate MSE loss for LSD and Affs
        lsd_loss = self._calc_loss(pred_lsds, gt_lsds, lsds_weights)
        aff_loss = self.aff_lambda * self._calc_loss(pred_affs, gt_affs, affs_weights)

        # calculate MSE loss for GAN errors, pass data through discrim network in process
        if gt_enhanced is not None and pred_enhanced is not None:
            real_scores = self.discriminator(gt_enhanced)
            fake_scores = self.discriminator(pred_enhanced)
            gan_loss = self.gan_lambda * (
                torch.mean((real_scores - 1) ** 2) + torch.mean(fake_scores**2)
            )
        else:
            gan_loss: float = 0.0

        return lsd_loss + aff_loss + gan_loss
