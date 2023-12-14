import torch


class WeightedACLSD_MSELoss(torch.nn.MSELoss):
    """
    Weighted Auto-Context LSD (ACLSD) Mean Squared Error (MSE) Loss.

    This loss is an extension of the traditional MSE loss with an additional
    weighting term for Auto-Context LSD (ACLSD) segmentation.

    Parameters:
        aff_lambda (float, optional): Weighting factor for the affinity loss. Default is 1.0.
    """

    def __init__(self, aff_lambda=1.0) -> None:
        """
        Initializes the WeightedACLSD_MSELoss.

        Args:
            aff_lambda (float, optional): Weighting factor for the affinity loss. Default is 1.0.
        """
        super(WeightedACLSD_MSELoss, self).__init__()

        self.aff_lambda = aff_lambda

    def _calc_loss(self, prediction, target, weights):
        """
        Calculates the weighted mean squared error loss.

        Args:
            prediction (torch.Tensor): Predicted affinities.
            target (torch.Tensor): Ground truth affinities.
            weights (torch.Tensor): Weighting factor for each affinity.

        Returns:
            torch.Tensor: Weighted mean squared error loss.
        """
        scaled = weights * (prediction - target) ** 2

        if len(torch.nonzero(scaled)) != 0:
            mask = torch.masked_select(scaled, torch.gt(weights, 0))
            loss = torch.mean(mask)
        else:
            loss = torch.mean(scaled)

        return loss

    def forward(
        self,
        pred_affs=None,
        gt_affs=None,
        affs_weights=None,
    ):
        """
        Calculates the weighted ACLSD MSE loss.

        Args:
            pred_affs (torch.Tensor): Predicted affinities.
            gt_affs (torch.Tensor): Ground truth affinities.
            affs_weights (torch.Tensor): Weighting factor for each affinity.

        Returns:
            torch.Tensor: Weighted ACLSD MSE loss.
        """
        aff_loss = self.aff_lambda * self._calc_loss(pred_affs, gt_affs, affs_weights)

        return aff_loss
