from .train import mtlsd_train, aclsd_train, stelarr_train

def train_model(
    model_type: str = "MTLSD",
    iterations: int = 100000,
    warmup: int = 100000,
    raw_file: str = "path/to/zarr/or/n5",
    voxel_size: int = 33,
) -> None:
    model_type = model_type.lower()
    if model_type == "mtlsd":
        mtlsd_train(iterations=iterations, raw_file=raw_file, voxel_size=voxel_size)
    elif model_type == "aclsd":
        aclsd_train(iterations=iterations, raw_file=raw_file, warmup=warmup)
    elif model_type == "stelarr":
        stelarr_train(iterations=iterations, raw_file=raw_file, warmup=warmup)
