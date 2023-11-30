from .train import mtlsd_train, aclsd_train, stelarr_train
from .utils import tiff_to_zarr


def train_model(
    model_type: str = "MTLSD",
    iterations: int = 100000,
    warmup: int = 100000,
    raw_file: str = "path/to/.zarr/or/.n5/or/.tiff",
    rewrite_file: str = "./rewritten.zarr",
    rewrite_ds: str = "rewritten_volume",
    out_file: str = "./raw_predictions.zarr",
    voxel_size: int = 33,
    save_every=2500,
) -> None:
    
    # TODO: add util funcs for generating masks 
    if raw_file.endswith(".tiff"):
        tiff_to_zarr(tiff_file=raw_file,
                     out_file=rewrite_file,
                     out_ds=rewrite_ds)
    

    model_type = model_type.lower()
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
