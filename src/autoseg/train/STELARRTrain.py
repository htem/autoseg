import gunpowder as gp
import logging
import math
import numpy as np
import torch
from lsd.train.gp import AddLocalShapeDescriptor
from tqdm import trange

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

from ..models.STELARRModel import STELARRModel
from ..postprocess.segment_skel_correct import get_skel_correct_segmentation
from ..networks.NLayerDiscriminator import NLayerDiscriminator, NLayerDiscriminator3D
from ..networks.FLibUNet import setup_unet
from ..losses.GANLoss import GANLoss
from ..losses.GMSELoss import Weighted_GMSELoss
from ..gp_filters.random_noise import RandomNoiseAugment
from ..gp_filters.smooth_array import SmoothArray
from ..utils import neighborhood


def stelarr_train(
    raw_file: str = "../../data/xpress-challenge.zarr",
    out_file: str = "./raw_predictions.zarr",
    voxel_size: int = 33,
    iterations: int = 100000,
    warmup: int = 200000,
    save_every: int = 25000,
) -> None:
    """
    Train STELARR model using Gunpowder library.

    Args:
        raw_file (str):
            Path to the raw data file.
        out_file (str):
            Path to the output file for raw predictions.
        voxel_size (int):
            Voxel size.
        iterations (int):
            Number of training iterations.
        warmup (int):
            Number of warm-up iterations.
        save_every (int):
            Save predictions every 'save_every' iterations.
    """
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    labels_mask = gp.ArrayKey("LABELS_MASK")
    unlabelled = gp.ArrayKey("UNLABELLED")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    gt_lsds_mask = gp.ArrayKey("GT_LSDS_MASK")
    pred_enhanced = gp.ArrayKey("PRED_ENHANCED")
    gt_enhanced = gp.ArrayKey("GT_ENHANCED")
    fake_pred = gp.ArrayKey("FAKE_PRED")
    real_pred = gp.ArrayKey("REAL_PRED")

    unet = setup_unet(downsample_factors=[(2, 2, 2), (2, 2, 2)], num_heads=3)
    model: STELARRModel = STELARRModel(unet=unet, num_fmaps=unet.out_channels)
    discriminator: NLayerDiscriminator3D = NLayerDiscriminator(
        ndims=3,
    )  # NLayerDiscriminator3D
    loss: Weighted_GMSELoss = Weighted_GMSELoss(discrim=discriminator)  # aff_lambda=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, betas=(0.95, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.5e-4, betas=(0.95, 0.999)
    )
    discriminator_loss: GANLoss = GANLoss()

    input_shape = [100] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]
    logging.info(input_shape, output_shape)

    voxel_size = gp.Coordinate((voxel_size,) * 3)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    # Zarr sources
    predicted_source = (
        gp.ZarrSource(
            raw_file,
            {raw: "volumes/training_raw", gt_enhanced: "volumes/training_gt_enhanced"},
            {
                raw: gp.ArraySpec(interpolatable=True),
                gt_enhanced: gp.ArraySpec(interpolatable=False),
            },
        ),
        gp.ZarrSource(
            out_file,
            {
                labels: f"pred_seg",
                labels_mask: f"pred_labels_mask",
                unlabelled: f"pred_unlabelled_mask",
            },
            {
                labels: gp.ArraySpec(interpolatable=False),
                labels_mask: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
            },
        ),
    ) + gp.MergeProvider()
    predicted_source += gp.MergeProvider()

    predicted_source += gp.RandomLocation(mask=labels_mask, min_masked=0.5)
    gt_source = gp.ZarrSource(
        raw_file,
        {
            raw: "volumes/training_raw",
            labels: f"volumes/training_gt_labels",
            labels_mask: f"volumes/training_labels_mask",
            unlabelled: f"volumes/training_unlabelled_mask",
            gt_enhanced: f"volumes/training_gt_enhanced",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            labels_mask: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
            gt_enhanced: gp.ArraySpec(interpolatable=False),
        },
    )
    gt_source += gp.MergeProvider()
    gt_source += gp.RandomLocation(mask=labels_mask, min_masked=0.5)

    def get_training_pipeline():
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
        request.add(pred_enhanced, output_size)
        request.add(gt_enhanced, input_size)

        training_pipeline = gp.Normalize(raw)
        training_pipeline += gp.Normalize(gt_enhanced)
        training_pipeline += gp.Pad(raw, None)
        training_pipeline += gp.Pad(gt_enhanced, context)
        training_pipeline += gp.Pad(labels, context)
        training_pipeline += gp.Pad(labels_mask, context)
        training_pipeline += gp.Pad(unlabelled, context)

        training_pipeline += gp.ElasticAugment(
            control_point_spacing=[30, 30, 30],
            jitter_sigma=[2, 2, 2],
            rotation_interval=[0, math.pi / 2.0],
            subsample=8,
        )

        training_pipeline += gp.SimpleAugment()

        training_pipeline += RandomNoiseAugment(raw)

        training_pipeline += gp.IntensityAugment(raw, 0.9, 1.1, -0.1, 0.1)

        training_pipeline += SmoothArray(raw, (0.0, 1.0))

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
            dtype=np.float32,
        )

        training_pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

        training_pipeline += gp.Unsqueeze([raw, gt_enhanced])
        training_pipeline += gp.Stack(1)

        training_pipeline += gp.PreCache(cache_size=40, num_workers=10)

        training_pipeline += gp.torch.Train(
            model,
            loss,
            optimizer,
            inputs={"input": raw},
            loss_inputs={
                0: pred_lsds,
                1: gt_lsds,
                2: gt_lsds_mask,
                3: pred_affs,
                4: gt_affs,
                5: affs_weights,
                6: pred_enhanced,
                7: gt_enhanced,
            },
            outputs={0: pred_lsds, 1: pred_affs, 2: pred_enhanced},
            save_every=save_every,
            checkpoint_basename="multitask_model",
            log_dir="log/mt_log",
        )

        # two train nodes, use GANloss
        training_pipeline += gp.torch.Train(
            discriminator,
            discriminator_loss,  # GAN Loss Y with lambda
            discriminator_optimizer,
            inputs={"input": gt_enhanced},
            loss_inputs={
                "real_pred": real_pred,
            },
            outputs={0: real_pred},
            save_every=save_every,
            checkpoint_basename="discrim_model",
            log_dir="log/discrim_log",
        )
        training_pipeline += gp.torch.Train(
            discriminator,
            discriminator_loss,  # GAN Loss X with lambda
            discriminator_optimizer,
            inputs={"input": pred_enhanced},
            loss_inputs={
                "fake_pred": fake_pred,
            },
            outputs={0: fake_pred},
            save_every=save_every,
            checkpoint_basename="discrim_model",
            log_dir="log/discrim_log",
        )

        training_pipeline += gp.Squeeze(
            [raw, gt_lsds, pred_lsds, gt_affs, pred_affs, gt_enhanced, pred_enhanced]
        )

        training_pipeline += gp.Snapshot(
            dataset_names={
                raw: "raw",
                labels: "labels",
                gt_lsds: "gt_lsds",
                unlabelled: "unlabelled",
                pred_lsds: "pred_lsds",
                gt_affs: "gt_affs",
                pred_affs: "pred_affs",
                affs_weights: "affs_weights",
                gt_enhanced: "gt_enhanced",
                pred_enhanced: "pred_enhanced",
            },
            dataset_dtypes={gt_affs: np.float32},
            output_filename="batch_latest.zarr",
            every=save_every,
        )

        return training_pipeline, request

    # First iterations are warmup on voxel data
    if (
        warmup is None
    ):  # Allows to do initial segmentation with existing model checkpoints
        # Make segmentation predictions
        get_skel_correct_segmentation(
            predict_affs=True,
            raw_file=raw_file,
            out_file=out_file,
            voxel_size=voxel_size,
        )
        model.train()
    elif warmup > 0:
        training_pipeline, request = get_training_pipeline()
        logging.info("PIPELINE IS SET . . .")
        logging.info(gt_source)
        logging.info(training_pipeline)
        pipeline = gt_source + training_pipeline

        with gp.build(pipeline):
            for i in trange(warmup):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_skel_correct_segmentation(
            predict_affs=True,
            raw_file=raw_file,
            out_file=out_file,
            voxel_size=voxel_size,
        )
        model.train()

    # Add segmentation predictions to training pipeline
    # Then repeat, scaling up the prediction usage
    for ratio in [0, 1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        print(f"Rinse & Repeat @ ratio: {ratio}")
        training_pipeline, request = get_training_pipeline()
        pipeline = (gt_source, predicted_source) + gp.RandomProvider(
            probabilities=[1 - ratio, ratio]
        )
        pipeline += training_pipeline
        with gp.build(pipeline):
            for i in trange(iterations):
                pipeline.request_batch(request)

        # Make segmentation predictions
        get_skel_correct_segmentation(
            predict_affs=True,
            raw_file=raw_file,
            out_file=out_file,
            voxel_size=voxel_size,
        )
        model.train()
