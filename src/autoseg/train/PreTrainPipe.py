import math
import gunpowder as gp
from lsd.train.gp import AddLocalShapeDescriptor
import numpy as numpy

from ..gp_filters.random_noise import RandomNoiseAugment
from ..gp_filters.smooth_array import SmoothArray


def pretrain_pipe(training_pipeline: gp.ZarrSource, ) -> gp.ZarrSource: # TODO: fix args
    training_pipeline += gp.ElasticAugment(
        control_point_spacing=[30, 30, 30],
        jitter_sigma=[2, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        subsample=8,
    )

    training_pipeline += gp.SimpleAugment()

    training_pipeline += RandomNoiseAugment(raw)

    training_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    training_pipeline += SmoothArray(raw, (0.0,1.0))

    training_pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        sigma=10 * 33,
        lsds_mask=gt_lsds_mask,
        unlabelled=unlabelled,
        downsample=1,
    )

    training_pipeline += gp.GrowBoundary(labels, mask=unlabelled)

    training_pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        labels_mask=labels_mask,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32
        )
    return training_pipeline