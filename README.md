# autoseg

Automated segmentation for large-scale biological datasets.

<div align="center">
  <!-- Lint Actions -->
  <a href="https://github.com/htem/autoseg/actions/workflows/black.yaml">
    <img src="https://github.com/htem/autoseg/actions/workflows/black.yaml/badge.svg"
      alt="Lint Actions Status" />
  </a>
  <!-- Loss Test Actions -->
  <a href="https://github.com/htem/autoseg/actions/workflows/loss_tests.yaml">
    <img src="https://github.com/htem/autoseg/actions/workflows/loss_tests.yaml/badge.svg"
      alt="Loss Tests Actions Status" />
  </a>
  <!-- Network Test Actions -->
  <a href="https://github.com/htem/autoseg/actions/workflows/network_tests.yaml">
    <img src="https://github.com/htem/autoseg/actions/workflows/network_tests.yaml/badge.svg"
      alt="Network Tests Actions Status" />
  </a>
  <!-- Model Test Actions -->
  <a href="https://github.com/htem/autoseg/actions/workflows/model_tests.yaml">
    <img src="https://github.com/htem/autoseg/actions/workflows/model_tests.yaml/badge.svg"
      alt="Model Tests Actions Status" />
  </a>
</div>





* Free software: Apache 2.0 License

### Installation
A complete install can be run with: 
```bash
bash install.sh
```

Or install-by-install as follows:

1. Install Rust and Cargo via RustUp:

```bash
curl https://sh.rustup.rs -sSf | sh
```


2. Install MongoDB:

```bash
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
```


3. And initialize a MongoDB server in a screen on your machine:

```bash
screen
mongod
```

4. Install ``graph_tool``

```bash
conda install -c conda-forge -c ostrokach-forge -c pkgw-forge graph-tool
```


5. Install `autoseg`:

```bash
pip install git+https://github.com/brianreicher/autoseg.git
```

### Features
TODO

### Usage
This package is used for training, predicting, & evaluating deep learning segmentation models. The models are compatible with Zarr & N5 chunked image files, and volumes should be stored in the following format:

```
your_dataset.zarr/
|-- volumes/
|   |-- training_raw/
|   |   |-- 0/
|   |   |   |-- <raw_data_chunk_0>
|   |   |   |-- <raw_data_chunk_1>
|   |   |   |   ...
|   |   |-- 1/
|   |       ...
|-- training_labels/
|   |-- 0/
|   |   |-- <label_chunk_0>
|   |   |-- <label_chunk_1>
|   |   |   ...
|   |-- 1/
|       ...
|-- training_labels_masked/
|   |-- 0/
|   |   |-- <masked_label_chunk_0>
|   |   |-- <masked_label_chunk_1>
|   |   |   ...
|   |-- 1/
|       ...
|-- training_labels_unmasked/
    |-- 0/
    |   |-- <unmasked_label_chunk_0>
    |   |-- <unmasked_label_chun
```

1. The first step in the `autoseg` segmentation pipeline is predicting pixel affinities. Pointing the `autoseg.train_model()` driver function to a zarr with the following volumes will allow for training, along with hyperparameter tuning.

```python
from autoseg import train_model


train_model(
    model_type="MTLSD",
    iterations=100000,
    warmup=100000,
    raw_file="path/to/your/raw/data.zarr",
    out_file="./raw_predictions.zarr",
    voxel_size=33,
    save_every=25000,
)
```
Functonality exists in `autoseg.utils` and `autoseg.WebknossosToolkit()` to handle data fetching, transformations, and conversions.

2. After affinities have been predicted, the `autoseg.postprocess` module is used to run Mutex Watershed or Merge Tree instance segmentation. Users can pass in the desired affinities Zarr to segment, as follows:
```python
from autoseg import postprocess.get_validation_segmentation


get_validation_segmentation(
    segmentation_style: str = "mws",
    iteration="latest",
    raw_file="./data.zarr",
    raw_dataset="volumes/validation_raw",
    out_file="./validation.zarr",
)
```
TODO

### Credits

This package builds upon segmentation pipelines & algorithms developed by Arlo Sheridan, Brian Reicher, Jeff Rhoades, Vijay Venu, and William Patton.
