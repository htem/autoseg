import gunpowder as gp
import logging
import numpy as np
import os
import glob
import torch
from funlib.persistence import prepare_ds
import daisy
from ..utils import neighborhood

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_task(
    model:torch.nn.Module,
    model_type: str,
    iteration: int,
    raw_file: str,
    raw_dataset: str,
    out_file: str = "raw_prediction.zarr",
    out_datasets: list[tuple] = [
        (f"pred_affs", len(neighborhood)),
        (f"pred_lsds", 10),
        (f"pred_enhanced", 1),
    ],
    num_workers: int = 1,
    n_gpu: int = 1,
    model_path: str = "./",
    voxel_size: int = 33,
) -> None:
    """
    Predict segmentation outputs using a trained deep learning model.

    Parameters:
        model: 
            Trained deep learning model.
        model_type (str): 
            Type of the model ("MTLSD", "ACLSD", "STELARR").
        iteration (int): 
            Iteration or checkpoint of the trained model to use.
        raw_file (str): 
            Path to the input Zarr or N5 dataset or TIFF file containing raw data.
        raw_dataset (str): 
            Name of the raw dataset in the input Zarr or N5 file.
        out_file (str): 
            Path to the output Zarr file for storing predictions.
        out_datasets (list): 
            List of tuples specifying output dataset names and channel counts.
        num_workers (int): 
            Number of parallel workers for blockwise processing.
        n_gpu (int): 
            Number of GPUs available for prediction.
        model_path (str): 
            Path to the directory containing the trained model checkpoints.
        voxel_size (int): 
            Voxel size in all three dimensions.

    Returns:
        None: 
            No return value. Predictions are stored in the specified Zarr file.
    """
    if type(iteration) == str and "latest" in iteration:
        model_path = glob.glob(
            os.path.join(model_path, f"{model_type}_model_checkpoint_*")
        )
        model_path.sort(key=os.path.getmtime)
        model_path = os.path.abspath(model_path[-1])
        print(f"Model path: {model_path}")

    else:
        model_path = os.path.abspath(
            os.path.join(model_path, f"{model_type}_model_checkpoint_{iteration}")
        )

    # input/output for multitask_model
    input_shape = [100] * 3
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[2:]
    logger.info((input_shape, output_shape))

    voxel_size = gp.Coordinate((voxel_size,) * 3)
    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    context = ((input_size - output_size) / 2) * 4

    raw = gp.ArrayKey("RAW")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    pred_enhanced = gp.ArrayKey("PRED_ENHANCED")

    source = gp.ZarrSource(
        raw_file, {raw: raw_dataset}, {raw: gp.ArraySpec(interpolatable=True)}
    )

    with gp.build(source):
        total_input_roi = source.spec[raw].roi
        total_output_roi = total_input_roi.grow(-context, -context)
        print(total_output_roi)
    for ds_name, channels in out_datasets:
        logger.info(f"Preparing {ds_name} with {channels} channels...")
        prepare_ds(
            out_file,
            ds_name,
            total_output_roi,
            voxel_size,
            dtype=np.uint8,
            num_channels=channels,
            write_size=output_size,
            compressor={"id": "blosc"},
            delete=True,
        )

    block_read_roi = daisy.Roi((0,) * 3, input_size) - context
    block_write_roi = daisy.Roi((0,) * 3, output_size)

    def predict():
        # set model to evaluation mode
        model.eval()

        scan_request = gp.BatchRequest()

        scan_request.add(raw, input_size)
        scan_request.add(pred_affs, output_size)
        scan_request.add(pred_lsds, output_size)
        outputs = {
            0: pred_lsds,
            1: pred_affs,
        }
        ds_names: dict = {
            pred_affs: out_datasets[0][0],
            pred_lsds: out_datasets[1][0],
        }
        daisy_request: dict = {
            raw: "read_roi",
            pred_lsds: "write_roi",
            pred_affs: "write_roi",
        }

        if "stelarr" in model_type.lower():
            scan_request.add(pred_enhanced, output_size)
            outputs: dict = {0: pred_lsds, 1: pred_affs, 2: pred_enhanced}
            ds_names: dict = {
                pred_affs: out_datasets[0][0],
                pred_lsds: out_datasets[1][0],
                pred_enhanced: out_datasets[2][0],
            }
            daisy_request: dict = {
                raw: "read_roi",
                pred_lsds: "write_roi",
                pred_affs: "write_roi",
                pred_enhanced: "write_roi",
            }

        # predict and write the initial pass
        pred = gp.torch.Predict(
            model,
            checkpoint=model_path,
            inputs={"input": raw},
            outputs=outputs,
        )

        write = gp.ZarrWrite(
            dataset_names=ds_names,
            output_filename=out_file,
        )

        if num_workers > 1:
            worker_id = int(daisy.Context.from_env()["worker_id"])
            logger.info(worker_id % n_gpu)
            os.environ["CUDA_VISISBLE_DEVICES"] = f"{worker_id % n_gpu}"

            scan = gp.DaisyRequestBlocks(
                scan_request,
                daisy_request,
                num_workers=num_workers,
            )

        else:
            scan = gp.Scan(scan_request)

        if "stelarr" in model_type.lower():
            pipeline = (
                source
                + gp.Normalize(raw)
                + gp.Unsqueeze([raw])
                + gp.Unsqueeze([raw])
                + pred
                + gp.Squeeze([pred_affs])
                + gp.Squeeze([pred_lsds])
                + gp.Squeeze([pred_enhanced])
                + gp.Normalize(pred_affs)
                + gp.Normalize(pred_enhanced)
                + gp.IntensityScaleShift(pred_affs, 255, 0)
                + gp.IntensityScaleShift(pred_lsds, 255, 0)
                + gp.IntensityScaleShift(pred_enhanced, 255, 0)
                + write
                + scan
            )
        else:
            pipeline = (
                source
                + gp.Normalize(raw)
                + gp.Unsqueeze([raw])
                + gp.Unsqueeze([raw])
                + pred
                + gp.Squeeze([pred_affs])
                + gp.Squeeze([pred_lsds])
                + gp.Normalize(pred_affs)
                + gp.IntensityScaleShift(pred_affs, 255, 0)
                + gp.IntensityScaleShift(pred_lsds, 255, 0)
                + write
                + scan
            )

        predict_request = gp.BatchRequest()

        if num_workers == 1:
            predict_request[raw] = total_input_roi
            predict_request[pred_affs] = total_output_roi

        with gp.build(pipeline):
            batch = pipeline.request_batch(predict_request)

    if num_workers > 1:
        task = daisy.Task(
            "PredictBlockwiseTask",
            total_input_roi,
            block_read_roi,
            block_write_roi,
            process_function=predict,
            num_workers=num_workers,
            max_retries=3,
            fit="shrink",
        )

        done: bool = daisy.run_blockwise([task])

        if not done:
            raise RuntimeError("at least one block failed!")

    else:
        predict()
