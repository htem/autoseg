# UTILIZES RUSTY_MWS, HGLOM FOR ALL SEGMENTATION, ORIGINALLY WRITTEN BY BRIAN REICHER (2023)

import rusty_mws
import hglom

from ..utils import neighborhood
from ..predict.network_predictions import predict_task


def get_validation_segmentation(
    segmentation_style:str="mws",
    iteration="latest",
    raw_file="../../data/xpress-challenge.zarr",
    raw_dataset="volumes/validation_raw",
    out_file="./validation.zarr",
    pred_affs=True
) -> bool:

    out_datasets = [
        (f"pred_affs_{iteration}", len(neighborhood)),
        (f"pred_lsds_{iteration}", 10),
        (f"pred_enhanced_{iteration}", 1),
    ]

    if pred_affs:
        predict_task( # Raw --> Affinities
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=out_datasets,
            num_workers=1,
            n_gpu=1)
    if segmentation_style.lower() == "mws":
        pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor( # Affinities -> Segmentation
                    affs_file=out_file,
                    affs_dataset=out_datasets[0][0],
                    seeds_file=raw_file,
                    seeds_dataset="volumes/validation_gt_rasters",
                    seg_dataset="segmentation_mws",
                    neighborhood_length=15,
                    n_chunk_write_frags=1,
                    filter_val=.6,
                    adjacent_edge_bias=.5,
                    lr_bias=.5,
                    adj_bias=-.4
                )
        success: bool = pp.segment_mws()
    elif segmentation_style.lower() == "mergetree":
        pp: hglom.PostProcessor = hglom.PostProcessor(
                    affs_file="./validation.zarr",
                    affs_dataset="pred_affs_latest",
                    fragments_dataset="frags_mergetree",
                    seg_dataset="seg_mergetree",
                    merge_function="hist_quant_50"
                )
        success: bool = pp.segment()
    
    return success

