from more_itertools import raise_
from .train import mtlsd_train, aclsd_train, stelarr_train
from .utils import tiff_to_zarr, create_masks


def train_model(
    model_type: str = "MTLSD",
    iterations: int = 100000,
    warmup: int = 100000,
    raw_file: str = "path/to/.zarr/or/.n5/or/.tiff",
    rewrite_file: str = "./rewritten.zarr",
    rewrite_ds: str = "volumes/training_raw",
    out_file: str = "./raw_predictions.zarr",
    generate_masks: bool = False,
    voxel_size: int = 33,
    save_every=2500,
) -> None:
    
    # TODO: add util funcs for generating masks, pulling paintings
    if raw_file.endswith(".tiff") or raw_file.endswith(".tif"):
        try:
            tiff_to_zarr(tiff_file=raw_file,
                        out_file=rewrite_file,
                        out_ds=rewrite_ds)
            raw_file: str = rewrite_file
        except:
            raise("Could not convert TIFF file to zarr volume")
    
    if generate_masks:
        try:
            create_masks(raw_file, "volumes/training_gt_labels")
        except:
            raise("Could not generate masks - check to make sure a painting labels volume exists")

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
