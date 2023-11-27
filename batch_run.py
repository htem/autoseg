import argparse
from autoseg import train_model


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
