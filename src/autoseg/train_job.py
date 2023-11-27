import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models")
    parser.add_argument("--model_type", type=str, default="MTLSD", choices=["MTLSD", "ACLSD", "STELARR"], help="Type of model to train")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=100000, help="Warmup value")
    parser.add_argument("--raw_file", type=str, default="path/to/zarr/or/n5", help="Path to raw file")
    parser.add_argument("--voxel_size", type=int, default=33, help="Voxel size")

    args = parser.parse_args()
    train_model(
        model_type=args.model_type,
        iterations=args.iterations,
        warmup=args.warmup,
        raw_file=args.raw_file,
        voxel_size=args.voxel_size,
    )
