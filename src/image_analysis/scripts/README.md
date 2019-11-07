# `Python` scripts

In this directory we keep `python` scripts that manipulate the output of the
image analysis pipeline to compute relevant quantities like error estimates on
mean or noise in gene expression.

- `export_microscopy_single_cell.py` : This script exports a tidy `dataframe`
  that contains all of the single-cell measurements that passed the quality
  control tests.
- `fc_noise_bootstrap.py` : This script computes bootstrap estimates of the
  fold-change and noise in gene expression estimates from the experimental
  data.