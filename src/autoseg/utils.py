import numpy as np
from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate, Roi
import tifffile
import numpy as np
import zarr
import numpy as np
import daisy
import zarr
from skimage.draw import line_nd


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

def generate_graph(test_array, skeleton_path):
    print("Loading from file . . .")
    gt_graph = np.load(skeleton_path, allow_pickle=True)
    print("Successfully loaded")
    nodes_outside_roi = []
    for i, (treenode, attr) in enumerate(gt_graph.nodes(data=True)):
        pos = attr["position"]
        attr["zyx_coord"] = (pos[2], pos[1], pos[0])

        if not test_array.roi.contains(daisy.Coordinate(attr["zyx_coord"])):
            nodes_outside_roi.append(treenode)

    for node in nodes_outside_roi:
        gt_graph.remove_node(node)

    return gt_graph


def create_array(test_array, gt_graph):
    gt_ndarray = np.zeros_like(test_array.data).astype(np.uint32)
    voxel_size = test_array.voxel_size
    offset = test_array.roi.offset

    for u, v in gt_graph.edges():
        # todo - don't hardcode voxel size and offset here
        source = [
            (i / vx) - o
            for i, vx, o in zip(gt_graph.nodes[u]["zyx_coord"], voxel_size, offset)
        ]
        target = [
            (i / vx) - o
            for i, vx, o in zip(gt_graph.nodes[v]["zyx_coord"], voxel_size, offset)
        ]

        line = line_nd(source, target)

        gt_ndarray[line] = gt_graph.nodes[u]["skeleton_id"]

    return gt_ndarray


def rasterized_skeletons(raw_file: str, raw_ds: str, out_file: str, skeleton_path: str) -> None:

    array = daisy.open_ds(raw_file, raw_ds)
    gt_graph = generate_graph(array, skeleton_path)
    gt_array = create_array(array, gt_graph)

    out = zarr.open(out_file, "a")
    unabelled_mask = (gt_array > 0).astype(np.uint8)

    out["volumes/validation_gt_rasters"] = gt_array
    out["volumes/validation_gt_rasters"].attrs[
        "resolution"
    ] = array.voxel_size
    out["volumes/validation_gt_rasters"].attrs["offset"] = array.roi.offset
