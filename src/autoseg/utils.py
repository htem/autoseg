import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import tifffile
import numpy as np
import zarr


neighborhood: list[list[int]] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [4, 0, 0],
    [0, 4, 0],
    [0, 0, 4],
    [8, 0, 0],
    [0, 8, 0],
    [0, 0, 8],
    [10, 0, 0],
    [0, 10, 0],
    [0, 0, 10],
]

def tiff_to_zarr(tiff_file:str="path/to/.tiff",
                out_file:str="tiffAsZarr.zarr",
                out_ds:str="volumes/raw",
                voxel_size: int = 33,
                offset: int = 0,
                dtype=np.uint8,
                transpose:bool=False) -> None:
    tiff_stack: np.ndarray = tifffile.imread(tiff_file)
    if transpose:
        tiff_stack = np.transpose(tiff_stack, (2, 1, 0))

    voxel_size: Coordinate = Coordinate((voxel_size)*3)
    roi: Roi = Roi(offset=(offset)*3, shape=tiff_stack.shape * np.array(voxel_size))

    print("Roi: ", roi)
    voxel_size: Coordinate = Coordinate(100, 100, 100)

    ds = prepare_ds(
        filename=out_file,
        ds_name=out_ds,
        total_roi=roi,
        voxel_size=voxel_size,
        dtype=dtype,
        delete=True,
    )

    ds[roi] = tiff_stack

    print("TIFF Image stack saved as Zarr dataset.")


def create_masks(raw_file: str, labels_ds: str) -> None:
    f = zarr.open(raw_file, "a")

    labels = f[labels_ds]
    offset = labels.attrs["offset"]
    resolution = labels.attrs["resolution"]

    labels = labels[:]

    labels_mask = np.ones_like(labels).astype(np.uint8)
    unlabelled_mask = (labels > 0).astype(np.uint8)

    for ds_name, data in [
        ("volumes/training_labels_cropped_mask", labels_mask),
        ("volumes/training_unlabelled_cropped_mask", unlabelled_mask),
    ]:
        f[ds_name] = data
        f[ds_name].attrs["offset"] = offset
        f[ds_name].attrs["resolution"] = resolution

    try:
        labels = f["volumes/training_gt_rasters"]
        offset = labels.attrs["offset"]
        resolution = labels.attrs["resolution"]

        labels = labels[:]

        labels_mask = np.ones_like(labels).astype(np.uint8)
        unlabelled_mask = (labels > 0).astype(np.uint8)

        for ds_name, data in [
            ("volumes/training_raster_mask", labels_mask),
            ("volumes/training_unrastered_mask", unlabelled_mask),
        ]:
            f[ds_name] = data
            f[ds_name].attrs["offset"] = offset
            f[ds_name].attrs["resolution"] = resolution

    except KeyError:
        pass
