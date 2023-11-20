# UTILIZES RUSTY_MWS FOR ALL SEGMETATION ORIGINALLY WRITTEN BY BRIAN REICHER (2023)

import rusty_mws

from ..utils import neighborhood
from ..predict.network_predictions import predict_task


def get_skel_correct_segmentation(
    predict_affs: bool = True,
    raw_file: str = "../../data/xpress-challenge.zarr",
    raw_dataset: str = "volumes/training_raw",
    out_file: str = "./raw_predictions.zarr",
    out_datasets=[(f"pred_affs_latest", len(neighborhood)), (f"pred_lsds_latest", 10), (f"pred_enhanced_latest", 1)],
    iteration="latest",
    model_path="./",
    voxel_size:int=100,
) -> None:
    if predict_affs:
        # predict affs
        predict_task(
            iteration=iteration,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_file=out_file,
            out_datasets=out_datasets,
            num_workers=1,
            multitask_model_path=model_path,
            voxel_size=voxel_size
        )

    # rusty mws + correction using skeletons
    pp: rusty_mws.PostProcessor = rusty_mws.PostProcessor(
            affs_file=out_file,
            affs_dataset=out_datasets[0][0],
            fragments_file=out_file,
            fragments_dataset="frag_seg",
            seeds_file=raw_file,
            seeds_dataset="volumes/training_gt_rasters",
            seg_dataset="pred_seg",
            n_chunk_write_frags=1,
            erode_iterations=1,
            neighborhood_length=15,
            filter_val=.6,
            adjacent_edge_bias=.5,
            lr_bias=.5,
            adj_bias=-.4
        )
    
    pp.segment_seed_correction()
