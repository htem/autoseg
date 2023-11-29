import gunpowder as gp
import logging
import math
import numpy as np
import torch
from tqdm import tqdm
from lsd.train.gp import AddLocalShapeDescriptor

from ..models.MTLSDModel import MTLSDModel
from ..networks.FLibUNet import setup_unet
from ..losses.MSELoss import Weighted_MSELoss
from ..gp_filters.random_noise import RandomNoiseAugment
from ..gp_filters.smooth_array import SmoothArray
from ..utils import neighborhood

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


def mtlsd_train(iterations: int, raw_file: str, voxel_size: int = 33):
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    # initial MTLSD UNet
    unet = setup_unet()

    mtlsd_model = MTLSDModel(unet=unet, num_fmaps=unet.out_channels)
    mtlsd_loss = Weighted_MSELoss()  # aff_lambda=0)
    mtlsd_optimizer = torch.optim.Adam(
        params=mtlsd_model.parameters(), lr=0.5e-4, betas=(0.95, 0.999)
    )

    increase = 8 * 3

    input_shape = [100] * 3
    output_shape = mtlsd_model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[
        2:
    ]
    print(input_shape, output_shape)

    voxel_size = gp.Coordinate((voxel_size,) * 3)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    request = gp.BatchRequest()

    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(labels_mask, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_lsds, output_size)
    request.add(affs_weights, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(gt_lsds_mask, output_size)
    request.add(unlabelled, output_size)
    request.add(pred_affs, output_size)
    request.add(pred_lsds, output_size)

    source = gp.ZarrSource(
        store=raw_file,
        datasets={
            raw: f"volumes/training_raw",
            labels: f"volumes/training_gt_labels",
            labels_mask: f"volumes/training_labels_mask",
            unlabelled: f"volumes/training_unlabelled_mask",
        },
        array_specs={
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
        },
    )

    source += gp.Normalize(raw)
    source += gp.Pad(raw, None)
    source += gp.Pad(labels, context)
    source += gp.Pad(labels_mask, context)
    source += gp.Pad(unlabelled, context)
    source += gp.RandomLocation(mask=labels_mask, min_masked=0.6)

    pipeline = source

    pipeline += gp.RandomProvider()

    pipeline += gp.ElasticAugment(
        control_point_spacing=[30, 30, 30],
        jitter_sigma=[2, 2, 2],
        rotation_interval=[0, math.pi / 2.0],
        subsample=8,
    )

    pipeline += gp.SimpleAugment()

    pipeline += RandomNoiseAugment(raw)

    pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

    pipeline += SmoothArray(raw, (0.6, 1.0))

    pipeline += AddLocalShapeDescriptor(
        segmentation=labels,
        descriptor=gt_lsds,
        sigma=10 * 33,
        lsds_mask=gt_lsds_mask,
        unlabelled=unlabelled,
        downsample=2,
    )

    pipeline += gp.GrowBoundary(labels, mask=unlabelled)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        labels_mask=labels_mask,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    pipeline += gp.Unsqueeze([raw])
    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(cache_size=40, num_workers=10)

    pipeline += gp.torch.Train(
        mtlsd_model,
        mtlsd_loss,
        mtlsd_optimizer,
        inputs={"input": raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: gt_lsds_mask,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        outputs={0: pred_lsds, 1: pred_affs},
        save_every=50000,
        log_dir="log",
    )

    pipeline += gp.Squeeze([raw, gt_lsds, pred_lsds, gt_affs, pred_affs])

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            labels: "labels",
            gt_lsds: "gt_lsds",
            unlabelled: "unlabelled",
            pred_lsds: "pred_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        every=50000,
    )

    with gp.build(pipeline):
        for i in tqdm(range(iterations)):
            pipeline.request_batch(request)
