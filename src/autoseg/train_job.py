from .train import mtlsd_train, aclsd_train, stelarr_train


def train_model(
    model_type: str = "MTLSD",
    iterations: int = 100000,
    warmup: int = 100000,
    raw_file: str = "path/to/zarr/or/n5",
    voxel_size: int = 33,
) -> None:
    match model_type.lower():
        case "mtlsd":
            mtlsd_train(iterations=iterations, raw_file=raw_file, voxel_size=voxel_size)
        case "aclsd":
            aclsd_train(iterations=iterations, raw_file=raw_file, warmup=warmup)
        case "stelarr":
            stelarr_train(iterations=iterations, raw_file=raw_file, warmup=warmup)
