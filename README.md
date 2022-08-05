# Kernel Phase Pipeline for JWST

This package provides a pipeline to process JWST kernel phase interferometry (KPI) data.
The code aims to replicate the interface of the JWST pipeline. It uses XARA[^1] to
extract kernel phase observables. Preprocessing steps are also implemented to
enable extraction directly from the official stage 2 imaging JWST pipeline.

## Installation

### Virtual environment
It is recommended to installt he pipeline in a new virutal environment either
with conda:

```
conda create -n <env-name> python
conda activate <env-name>
```

or venv:

```
python -m venv <env-name>
source <env-name>/bin/activate
```

### Install jwst-kpi

<!-- TODO: Update path to Jens' repo -->

To install the jwst `KPI3Pipeline` in your new virtual environment, you can either clone and install locally:

```
git clone https://github.com/vandalt/jwst-kpi
cd jwst-kpi
python -m pip install .
```

or directly with `pip` from the Github repository:

```
python -m pip install git+https://github.com/vandalt/jwst-kpi
```


[^1]: The main XARA version is hosted here: https://github.com/fmartinache/xara.
  At the time of writing this, the pipeline uses a forked version of XARA
  (https://github.com/kammerje/xara/tree/develop) that implements new
  functionality required for the JWST pipeline.


## Usage

[An example](examples/niriss_kerphase.ipynb) of how to use the pipeline is
available. It shows how to go from uncalibrated JWST/NIRISS images to detection
limits and binary parameters.

### Pupil model

A discrete representation of the pupil is required from kernel phase. Default
pupil models for all supported instruments and modes are provided with the
package, but users can generate their own pupil models with XARA and use them in
the pipeline. We provide [a script](examples/generate_pupil_model.py) showing
how the default pupil models were generated. It can be use to generate new pupil
models. If you find a pupil model that performs better than the default one,
feel free to open an Issue or a PR!
