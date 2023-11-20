import torch


class WeightedACLSD_MSELoss(torch.nn.MSELoss):
    def __init__(self, aff_lambda=1.0) -> None:
        super(WeightedACLSD_MSELoss, self).__init__()

        self.aff_lambda = aff_lambda

    def _calc_loss(self, prediction, target, weights):
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
        aff_loss = self.aff_lambda * self._calc_loss(pred_affs, gt_affs, affs_weights)

        return aff_loss