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


### Usage


### Credits

This package builds upon segmentation pipelines & algorithms developed by Arlo Sheridan, Brian Reicher, Jeff Rhoades, Vijay Venu, and William Patton.
