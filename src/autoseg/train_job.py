from more_itertools import raise_
from .train import mtlsd_train, aclsd_train, stelarr_train
from .utils import tiff_to_zarr, create_masks
from .train_job import WebknossosToolkit


def train_model(
    model_type: str = "MTLSD",
    iterations: int = 100000,
    warmup: int = 100000,
    raw_file: str = "path/to/.zarr/or/.n5/or/.tiff",
    rewrite_file: str = "./rewritten.zarr",
    rewrite_ds: str = "volumes/training_raw",
    out_file: str = "./raw_predictions.zarr",
    get_labels: bool = False,
    get_rasters: bool = False,
    generate_masks: bool = False,
    voxel_size: int = 33,
    save_every: int = 2500,
    annotation_id: str = None,
    wk_token="YqSgxzFJpP2eyjtqymCTPg",
) -> None:
    # TODO: add util funcs for generating masks, pulling paintings
    if raw_file.endswith(".tiff") or raw_file.endswith(".tif"):
        try:
            tiff_to_zarr(tiff_file=raw_file, out_file=rewrite_file, out_ds=rewrite_ds)
            raw_file: str = rewrite_file
        except:
            raise ("Could not convert TIFF file to zarr volume")
    wk: WebknossosToolkit = WebknossosToolkit()

    if get_labels:
        try:
            wk.wkw_seg_to_zarr(
                annotation_id=annotation_id,
                save_path=".",
                zarr_path=raw_file,
                wk_token=wk_token,
                gt_name="training_labels",
            )
        except:
            raise ("Could not fetch and convert paintings to zarr format")

    if get_rasters:
        try:
            zip_path: str = wk.download_wk_skeleton(
                annotation_id=annotation_id, token=wk_token
            )
            wk.rasterize_skeleton(zip_path=zip_path, raw_file=raw_file)
        except:
            raise ("Could not fetch and convert skeletons to zarr format")

    if generate_masks:
        try:
            create_masks(raw_file, "volumes/training_gt_labels")
        except:
            raise (
                "Could not generate masks - check to make sure a painting labels volume exists"
            )

    model_type: str = model_type.lower()
    if model_type == "mtlsd":
        mtlsd_train(
            iterations=iterations,
            raw_file=raw_file,
            voxel_size=voxel_size,
            save_every=save_every,
        )
    elif model_type == "aclsd":
        aclsd_train(
            iterations=iterations,
            raw_file=raw_file,
            out_file=out_file,
            voxel_size=voxel_size,
            warmup=warmup,
            save_every=save_every,
        )
    elif model_type == "stelarr":
        stelarr_train(
            iterations=iterations,
            raw_file=raw_file,
            out_file=out_file,
            voxel_size=voxel_size,
            warmup=warmup,
            save_every=save_every,
        )
