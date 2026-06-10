# Probabilistic Deep Learning for WNPSH

A probabilistic deep learning framework for 1-month prediction of the Western North Pacific Subtropical High (WNPSH).

## Overview

This repository contains the code used for probabilistic prediction experiments of the Western North Pacific Subtropical High (WNPSH). 

If this repository is associated with a paper under review, the corresponding citation information will be updated after the review process.


## Requirements

The code has been tested with the following main packages:

- python
- pandas
- scikit-learn
- torch
- captum
- numpy
- netCDF4
- scipy
- matplotlib
- cartopy

## Installation

We recommend using a conda environment:

```bash
conda create -n wnpsh-dl python=3.9.23
conda activate wnpsh-dl
pip install -r requirements.txt
```


## Data

The datasets used in this study are not included in this repository. \
They will be made available via Zenodo upon publication. \
Please place the downloaded data in the appropriate directory before running the code.


## Notes
- This code may require a large amount of memory depending on the experiment settings and data size.
- Running the full workflow on a laptop may be impractical.
- GPU usage is recommended for training the deep learning model, although inference and evaluation can be performed on CPU if necessary.
- In our tests, the training workflow ran on a single NVIDIA GH100 GPU. This is provided as a reference environment, not as a strict requirement.
- The authors are not responsible for any issues or damages arising from the use of this repository.

## Reproducibility

To improve reproducibility, we recommend providing:

- exact package versions in `requirements.txt`
- random seed settings
- configuration files for training and evaluation
- a brief description of the input data format

## Authors

- Yuki Maeda
- Masaki Satoh

Affiliation: Atmosphere and Ocean Research Institute, The University of Tokyo

## Contact

For questions regarding this repository, please contact:

- Yuki Maeda  
  [maeda-y@aori.u-tokyo.ac.jp](mailto:maeda-y@aori.u-tokyo.ac.jp)

## License

This repository is released under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

