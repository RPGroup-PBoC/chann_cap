# `Python` scripts

This directory contains `pythons` scripts derived from the computations in the
`theory` notebooks. While in the notebooks we explain the logic and coding of
all the computations, in here we store scripts that use such computations
heavily. For example, when computing certain quantities for a myriad of
parameter values, instead of having the code in the notebook, we limit perform
such computations here. These scripts are grouped by the notebook from which
they derive.

### `moment_dynamics_cell_division.ipynb` (`mdcd`)

- `mdcd_iptg_range.py` : This script computes in parallel the average moments
  of the mRNA and protein distribution for a fine grid of IPTG values with the
  experimentally explored repressor copy numbers only.

- `mdcd_repressor_range.py` : This script computes in parallel the average
  moments of the mRNA and protein distribution for a fine grid of repressor
  copy number values with the 12 experimental IPTG concentrations.

