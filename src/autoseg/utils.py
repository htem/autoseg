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
import webknossos as wk
import wkw
from time import gmtime, strftime
import zipfile
import daisy
import tempfile
from glob import glob
import os
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds, prepare_ds
import numpy as np
import logging
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


def tiff_to_zarr(
    tiff_file: str = "path/to/.tiff",
    out_file: str = "tiffAsZarr.zarr",
    out_ds: str = "volumes/raw",
    voxel_size: int = 33,
    offset: int = 0,
    dtype=np.uint8,
    transpose: bool = False,
) -> None:
    tiff_stack: np.ndarray = tifffile.imread(tiff_file)
    if transpose:
        tiff_stack = np.transpose(tiff_stack, (2, 1, 0))

    voxel_size: Coordinate = Coordinate((voxel_size) * 3)
    roi: Roi = Roi(offset=(offset) * 3, shape=tiff_stack.shape * np.array(voxel_size))

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


def rasterized_skeletons(
    raw_file: str, raw_ds: str, out_file: str, skeleton_path: str
) -> None:
    array = daisy.open_ds(raw_file, raw_ds)
    gt_graph = generate_graph(array, skeleton_path)
    gt_array = create_array(array, gt_graph)

    out = zarr.open(out_file, "a")
    unabelled_mask = (gt_array > 0).astype(np.uint8)

    out["volumes/validation_gt_rasters"] = gt_array
    out["volumes/validation_gt_rasters"].attrs["resolution"] = array.voxel_size
    out["volumes/validation_gt_rasters"].attrs["offset"] = array.roi.offset


logger = logging.getLogger(__name__)


def download_wk_skeleton(
    save_path=".",
    url="http://catmaid2.hms.harvard.edu:9000",
    annotation_id=None,
    token=None,
    overwrite=True,
    zip_suffix=None,
):
    # print(f"Downloading {wk_url}/annotations/Explorational/{annotation_ID}...")
    with wk.webknossos_context(token=token, url=url):
        annotation = wk.Annotation.download(
            annotation_id,
            annotation_type="Explorational",
        )

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        if overwrite is None:
            overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite is True or overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            if zip_suffix is None:
                zip_suffix = (
                    f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
                )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)
    return zip_path


def parse_skeleton(zip_path):
    fin = zip_path
    if not fin.endswith(".zip"):
        try:
            fin = get_updated_skeleton(zip_path)
            assert fin.endswith(".zip"), "Skeleton zip file not found."
        except:
            assert False, "CATMAID NOT IMPLEMENTED"

    wk_skels = wk.skeleton.Skeleton.load(fin)
    # return wk_skels

    skel_coor = {}
    for tree in wk_skels.trees:
        skel_coor[tree.id] = []
        for start, end in tree.edges.keys():
            start_pos = start.position.to_np()
            end_pos = end.position.to_np()
            skel_coor[tree.id].append([start_pos, end_pos])

    return skel_coor


def get_updated_skeleton(zip_path):
    if not os.path.exists(zip_path):
        path = os.path.dirname(os.path.realpath(zip_path))
        search_path = os.path.join(path, "skeletons/*")
        files = glob(search_path)
        if len(files) == 0:
            skel_file = download_wk_skeleton()
        else:
            skel_file = max(files, key=os.path.getctime)
    skel_file = os.path.abspath(skel_file)

    return skel_file


def rasterize_skeleton(
    zip_path="/n/groups/htem/users/br128/xray-challenge-entry/monkeyv1axonseg001_KevinOhgami_20231010.zip",
    raw_file="./data/monkey_xnh.zarr",
    raw_ds="volumes/training_raw",
):
    logger.info(f"Rasterizing skeleton...")

    skel_coor = parse_skeleton(zip_path)

    # Initialize rasterized skeleton image
    raw = open_ds(raw_file, raw_ds)

    dataset_shape = raw.data.shape
    print(dataset_shape)
    voxel_size = raw.voxel_size
    offset = raw.roi.begin  # unhardcode for nonzero offset
    image = np.zeros(dataset_shape, dtype=np.uint8)

    def adjust(coor):
        ds_under = [x - 1 for x in dataset_shape]
        return np.min([coor - offset, ds_under], 0)

    print("adjusting . . .")
    for id, tree in skel_coor.items():
        # iterates through ever node and assigns id to {image}
        for start, end in tree:
            line = line_nd(adjust(start), adjust(end))
            image[line] = id

    # Save GT rasterization #TODO: implement daisy blockwise option
    total_roi = Roi(
        Coordinate(offset) * Coordinate(voxel_size),
        Coordinate(dataset_shape) * Coordinate(voxel_size),
    )

    print("saving . . .")
    out_ds = prepare_ds(
        raw_file,
        "volumes/training_rasters",
        total_roi,
        voxel_size,
        image.dtype,
        delete=True,
    )
    out_ds[out_ds.roi] = image

    return image


def get_wk_mask(
    annotation_ID,
    save_path,
    zarr_path,
    raw_name,
    wk_url="http://catmaid2.hms.harvard.edu:9000",
    wk_token="Q9OpWh1PPwHYfH9BsnoM2Q",
    save_name=None,
    mask_out=True,
):
    print(f"Downloading {wk_url}/annotations/Explorational/{annotation_ID}...")
    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(
            annotation_ID, annotation_type="Explorational"
        )

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            zip_suffix = input(
                f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
            )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)

    # Extract zip file
    zf = zipfile.ZipFile(zip_path)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        zipped_datafile = glob(tempdir + "/data*.zip")[0]
        print(f"Opening {zipped_datafile}...")
        zf_data = zipfile.ZipFile(zipped_datafile)
        with tempfile.TemporaryDirectory() as zf_data_tmpdir:
            zf_data.extractall(zf_data_tmpdir)

            # Open the WKW dataset (as the `1` folder)
            print(f"Opening {zf_data_tmpdir + '/1'}...")
            dataset = wkw.Dataset.open(zf_data_tmpdir + "/Volume/1")
            zarr_path = zarr_path.rstrip(os.sep)
            print(f"Opening {zarr_path}/{raw_name}...")
            ds = daisy.open_ds(zarr_path, raw_name)
            data = dataset.read(
                ds.roi.get_offset() / ds.voxel_size, ds.roi.get_shape() / ds.voxel_size
            ).squeeze()

    if save_name is not None:
        print("Saving...")
        target_roi = ds.roi
        if mask_out:
            mask_array = daisy.Array(data == 0, ds.roi, ds.voxel_size)
        else:
            mask_array = daisy.Array(data > 0, ds.roi, ds.voxel_size)

        chunk_size = ds.chunk_shape[0]
        num_channels = 1
        compressor = {
            "id": "blosc",
            "clevel": 3,
            "cname": "blosclz",
            "blocksize": chunk_size,
        }
        num_workers = 30
        write_size = mask_array.voxel_size * chunk_size
        chunk_roi = daisy.Roi(
            [
                0,
            ]
            * len(target_roi.get_offset()),
            write_size,
        )

        destination = daisy.prepare_ds(
            zarr_path,
            save_name,
            target_roi,
            mask_array.voxel_size,
            bool,
            write_size=write_size,
            write_roi=chunk_roi,
            # num_channels=num_channels,
            compressor=compressor,
        )

        # Prepare saving function/variables
        def save_chunk(block: daisy.Roi):
            destination.__setitem__(
                block.write_roi, mask_array.__getitem__(block.read_roi)
            )

        # Write data to new dataset
        task = daisy.Task(
            f"save>{save_name}",
            target_roi,
            chunk_roi,
            chunk_roi,
            process_function=save_chunk,
            read_write_conflict=False,
            fit="shrink",
            num_workers=num_workers,
            max_retries=2,
        )
        success = daisy.run_blockwise([task])

        if success:
            print(
                f"{target_roi} from {annotation_name} written to {zarr_path}/{save_name}"
            )
            return destination
        else:
            print("Failed to save annotation layer.")

    else:
        if mask_out:
            return daisy.Array(data == 0, ds.roi, ds.voxel_size)
        else:
            return daisy.Array(data > 0, ds.roi, ds.voxel_size)


# Extracts and saves volume annotations as a uint32 layer alongside the zarr used for making GT (>> assumes same ROI)
def wkw_seg_to_zarr(
    annotation_id,
    save_path,
    zarr_path,
    raw_name="volumes/training_raw",
    wk_url="http://catmaid2.hms.harvard.edu:9000",
    wk_token="YqSgxzFJpP2eyjtqymCTPg",
    gt_name=None,
    gt_name_prefix="volumes/",
    overwrite=None,
):
    print(f"Downloading {annotation_id} from {wk_url}...")
    with wk.webknossos_context(token=wk_token, url=wk_url):
        annotation = wk.Annotation.download(annotation_id)

    time_str = strftime("%Y%m%d", gmtime())
    annotation_name = (
        f'{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'
    )
    if save_path[-1] != os.sep:
        save_path += os.sep
    zip_path = save_path + annotation_name + ".zip"
    print(f"Saving as {zip_path}...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(zip_path):
        if overwrite is None:
            overwrite = input(f"{zip_path} already exists. Overwrite it? (y/n)")
        if overwrite.lower() == "y":
            os.remove(zip_path)
        else:
            zip_suffix = input(
                f"Save with new suffix? (Enter suffix, or leave blank to abort.)"
            )
            if zip_suffix != "":
                zip_path = save_path + annotation_name + "_" + zip_suffix + ".zip"
            else:
                print("Aborting...")
    annotation.save(zip_path)

    zarr_path = zarr_path.rstrip(os.sep)
    print(f"Opening {zarr_path}/{raw_name}...")
    # ds = daisy.open_ds(zarr_path, raw_name)
    # offset = ds.roi.get_offset() #/ ds.voxel_size
    # offset = Coordinate(0, 0, 0)
    roi = Roi((12600, 14100, 51100), (20000, 20000, 20000))
    shape = Coordinate(200, 200, 200)
    # shape = ds.roi.get_shape() / ds.voxel_size
    # shape = Roi()

    # Extract zip file
    zf = zipfile.ZipFile(zip_path)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        zipped_datafile = glob(tempdir + "/data*.zip")[0]
        print(f"Opening {zipped_datafile}...")
        zf_data = zipfile.ZipFile(zipped_datafile)
        with tempfile.TemporaryDirectory() as zf_data_tmpdir:
            zf_data.extractall(zf_data_tmpdir)

            # Open the WKW dataset (as the `1` folder)
            print(f"Opening {zf_data_tmpdir + '/1'}...")
            dataset = wkw.wkw.Dataset.open(zf_data_tmpdir + "/1")
            data = dataset.read(off=(126, 141, 511), shape=shape).squeeze()

    print(f"Sum of all data: {data.sum()}")
    # Save annotations to zarr
    if gt_name is None:
        gt_name = f'{gt_name_prefix}gt_{annotation.dataset_name}_{annotation.username.replace(" ","")}_{time_str}'

    target_roi = roi
    gt_array = daisy.Array(data, roi, (100, 100, 100))

    chunk_size = 1000
    num_channels = 1
    compressor = {
        "id": "blosc",
        "clevel": 3,
        "cname": "blosclz",
        "blocksize": chunk_size,
    }
    num_workers = 30
    write_size = gt_array.voxel_size * chunk_size
    chunk_roi = daisy.Roi(
        [
            0,
        ]
        * len(target_roi.get_offset()),
        write_size,
    )

    destination = daisy.prepare_ds(
        zarr_path,
        gt_name,
        target_roi,
        gt_array.voxel_size,
        data.dtype,
        write_size=write_size,
        delete=True,
    )

    # Prepare saving function/variables
    def save_chunk(block: daisy.Roi):
        try:
            destination.__setitem__(
                block.write_roi, gt_array.__getitem__(block.read_roi)
            )
            # destination[block.write_roi] = gt_array[block.read_roi]
            return 0  # success
        except:
            return 1  # error

    # Write data to new dataset
    task = daisy.Task(
        f"save>{gt_name}",
        target_roi,
        chunk_roi,
        chunk_roi,
        process_function=save_chunk,
        read_write_conflict=False,
        fit="shrink",
        num_workers=num_workers,
        max_retries=2,
    )
    success = daisy.run_blockwise([task])

    if success:
        print(f"{target_roi} from {annotation_name} written to {zarr_path}/{gt_name}")
        return gt_name
    else:
        print("Failed to save annotation layer.")
