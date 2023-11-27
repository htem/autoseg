from collections import defaultdict
import os
import daisy

import logging

logger: logging.Logger = logging.getLogger(__name__)

from evaluate import run_eval
from eval_db import Database
from ..postprocess import get_validation_segmentation


def segment_and_validate(
    model_checkpoint="latest",
    checkpoint_num=250000,
    setup_num="1738",
) -> dict:
    logger.info(
        msg=f"Segmenting checkpoint {model_checkpoint}, aff_model checkpoint {checkpoint_num}..."
    )

    success: bool = get_validation_segmentation(iteration=checkpoint_num)
    if success:
        print(
            "-----------------------------\nSuccessfully returned validation segmentation . . . now validating\n----------------------------------------------"
        )
        try:
            logger.info(
                f"Validating checkpoint {model_checkpoint}, aff_model checkpoint {checkpoint_num}..."
            )
            score_dict: dict = validate(
                checkpoint=model_checkpoint,
                threshold=float(f"{checkpoint_num}.{setup_num}"),
                ds="segmentation_mws",
            )
            logger.info(
                f"Validation for checkpoint {model_checkpoint}, aff_model checkpoint {checkpoint_num} successful"
            )
            return score_dict
        except Exception as e:
            logger.warn(
                f"Validation for checkpoint {model_checkpoint}, aff_model checkpoint {checkpoint_num} failed: {e}"
            )
    else:
        logger.warn(
            f"Validation for checkpoint {model_checkpoint}, aff_model checkpoint {checkpoint_num} failed"
        )

    return {}


def validate(
    checkpoint,
    threshold,
    offset: str = "3960,3960,3960",
    roi_shape: str = "31680,31680,31680",
    skel="../../data/XPRESS_validation_skels.npz",
    zarr="./validation.zarr",
    h5="validation.h5",
    ds="pred_seg",
    print_errors=False,
    print_in_xyz=False,
    downsample=None,
) -> None:
    network = os.path.abspath(".").split(os.path.sep)[-1]
    aff_setup, aff_checkpoint = str(threshold).split(".")[::-1]

    logger.info(f"Preparing {ds}")
    cmd_str = f"python ../../data/convert_to_zarr_h5.py {zarr} {ds} {h5} {ds}"
    if downsample is not None:
        cmd_str += f" --downsample {downsample}"
    os.system(cmd_str)

    # roi_begin = "8316,8316,8316"
    roi_begin = offset
    # roi_shape = "23067,23067,23067"
    roi_shape = roi_shape
    roi_begin = [float(k) for k in roi_begin.split(",")]
    roi_shape = [float(k) for k in roi_shape.split(",")]
    roi = daisy.Roi(roi_begin, roi_shape)

    logger.info(
        f"Evaluating {ds} for network {network}, checkpoint {checkpoint}, Raw->AFF setup{aff_setup}, checkpoint {aff_checkpoint}"
    )
    score_dict = run_eval(skel, h5, ds, roi, downsampling=downsample)
    logger.info(
        f"Finished evaluating {ds} for network {network}, checkpoint {checkpoint}. Saving results..."
    )

    split_edges = score_dict.pop("split_edges")
    merged_edges = score_dict.pop("merged_edges")
    gt_graph = score_dict.pop("gt_graph")

    try:
        db = Database("validation_results")
        db.add_score(
            network, checkpoint, threshold, score_dict
        )  # threshold is set as {checkpoint of LSD>AFF model}.{model number}
    except:
        pass

    # Terminal outputs
    logger.info(f'n_neurons: {score_dict["n_neurons"]}')
    logger.info(f'Expected run-length: {score_dict["erl"]}')
    logger.info(f'Normalized ERL: {score_dict["erl_norm"]}')

    logger.info("Count results:")
    logger.info(
        f'\tSplit count (total, per-neuron): {len(split_edges)}, {len(split_edges)/score_dict["n_neurons"]}'
    )
    logger.info(
        f'\tMerge count (total, per-neuron): {len(merged_edges)}, {len(merged_edges)/score_dict["n_neurons"]}'
    )

    if print_errors:
        gt_graph = score_dict["gt_graph"]

        def print_coords(node1, node2):
            node1_coord = daisy.Coordinate(gt_graph.nodes[node1]["zyx_coord"]) / 33
            node2_coord = daisy.Coordinate(gt_graph.nodes[node2]["zyx_coord"]) / 33
            if print_in_xyz:
                node1_coord = node1_coord[::-1]
                node2_coord = node2_coord[::-1]
            logger.info(f"{node1_coord} to {node2_coord}")

        logger.info("Split errors:")
        splits_by_skel = defaultdict(list)
        for edge in split_edges:
            splits_by_skel[gt_graph.nodes[edge[0]]["skeleton_id"]].append(edge)
        for skel in splits_by_skel:
            logger.info(f"Skeleton #{skel}")
            for edge in splits_by_skel[skel]:
                print_coords(edge[0], edge[1])
        logger.info("Split error histogram:")
        split_histogram = defaultdict(int)
        for i in range(score_dict["n_neurons"]):
            split_histogram[len(splits_by_skel[i])] += 1
        for k in sorted(split_histogram):
            logger.info(f"{k}: {split_histogram[k]}")

        logger.info("Merge errors:")
        for node1, node2 in merged_edges:
            print_coords(node1, node2)

    rand_voi = score_dict["rand_voi"]
    logger.info("Rand results (higher better):")
    logger.info(f"\tRand split: {rand_voi['rand_split']}")
    logger.info(f"\tRand merge: {rand_voi['rand_merge']}")
    logger.info("VOI results (lower better):")
    logger.info(f"\tNormalized VOI split: {rand_voi['nvi_split']}")
    logger.info(f"\tNormalized VOI merge: {rand_voi['nvi_merge']}")

    logger.info("XPRESS score (higher is better):")
    logger.info(f"\tERL+VOI : {score_dict['xpress_erl_voi']}")
    logger.info(f"\tERL+RAND: {score_dict['xpress_erl_rand']}")
    logger.info(f"\tVOI     : {score_dict['xpress_voi']}")
    logger.info(f"\tRAND    : {score_dict['xpress_rand']}")
    return score_dict
