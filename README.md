# JWST stage 3 pipeline for kernel phase imaging.

Authors: Jens Kammerer, Thomas Vandal, Katherine Thibault, Frantz Martinache

Supported instruments: NIRCam, NIRISS, MIRI

This package provides a pipeline to extract kernel phases from JWST data. The code aims to replicate the interface of the official JWST data reduction pipeline to provide a custom stage 3 kernel phase pipeline that outputs [KPFITS](https://ui.adsabs.harvard.edu/abs/2022arXiv221017528K/abstract) files. The pipeline is based on the XARA[^1] package and uses stage 2 calibrated ("cal" or "calints") products from the official JWST data reduction pipeline. 

## Installation

### Virtual environment

It is recommended to install the pipeline in a new virtual environment either with conda:

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

To install the `Kpi3Pipeline` in the new virtual environment, you can either clone and install locally:

```
git clone https://github.com/kammerje/jwst-kpi
cd jwst-kpi/
python -m pip install .
```

or directly with `pip` from the GitHub repository:

```
python -m pip install git+https://github.com/kammerje/jwst-kpi
```

**NOTE: With the development version, if you have errors about datamodels, please re-install the package with pip**

## Usage

Examples of how to use the pipeline are
available. There are examples for processing [NIRCam](examples/test_kpi_nircam.py), [NIRISS](examples/test_kpi_niriss.py), and [MIRI](examples/test_kpi_miri.py) cal and calints data.

### Pupil model

A discrete representation of the pupil is required for the kernel phase extraction. Default
pupil models for all supported instruments are provided with the
package, but users can generate their own pupil models with [XARA](https://github.com/fmartinache/xara) or [XAOSIM](https://github.com/fmartinache/xaosim) and use them in
the pipeline. We provide a [script](examples/generate_pupil_model.py) showing
how the default pupil models were generated. It can be used and adapted to generate custom pupil
models. If you find a pupil model that performs better than the default one,
feel free to open an Issue or a PR!

### News

The most recent major update brings new functionality and simplified usage:
* Both 2D cal and 3D calints data can now be processed.
* There is a new trim frames step. It is possible to specify the center and the size of the trimmed frames. With this new step, the pipeline is now running the steps in the following order: trim frames, fix bad pixels, recenter frames, window frames, extract kerphase, empirical uncertainties.
* It is now possible to provide a list of good frames and extract the kernel phase only from those.
* Improved pupil models and new pupil models for the NIRCam coronagraphy Lyot stops.
* Improved diagnostic plots.
* Improved file outputs.
* Simplification of the code by always transforming input data to 3D calints data.

[^1]: The main XARA version is hosted here: https://github.com/fmartinache/xara. At the time of writing this, the pipeline uses a forked version of XARA (https://github.com/kammerje/xara/tree/develop) that implements new functionality required for the JWST pipeline.
