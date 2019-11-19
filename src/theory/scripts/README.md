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

- `mdcd_repressor_extended_range.py` : This script computes in parallel the average
  moments of the mRNA and protein distribution for a grid of repressor up to 10^6
  copy number values with the 12 experimental IPTG concentrations.

- `mdcd_ogorman_param.py` : This script computes in parallel the average
  moments of the mRNA and protein distribution for the experimentally measured
  combinations of operators and repressors, but this time using the global
  parameter inferences as reported in [Chure et. al, 2019](https://www.rpgroup.caltech.edu/mwc_mutants/index.html)
  that phenomenologically capture better the induction profile for the O3
  operator and the general steepness of the other strains.

### `MaxEnt_approx_joint.ipynb` (`maxent`)

- `maxent_protein_dist.py` : Script that takes the protein distribution moments
  as inferred from the numerical integration of the dynamical equations and
  computes the corresponding Lagrange multipliers for a maximum entropy
  approximation of the distribution.

- `maxent_mRNA_dist.py` : Script that takes the mRNA distribution moments as
  inferred from the numerical integration of the dynamical equations and
  computes the corresponding Lagrange multipliers for a maximum entropy
  approximation of the distribution.

- `maxent_protein_dist_rep_range.py` : Script that takes the protein
  distribution moments as inferred from the numerical integration of the
  dynamical equations and computes the corresponding Lagrange multipliers for a
  maximum entropy approximation of the distribution for a larger span of
  repressor copy numbers.

- `maxent_protein_dist_iptg_range.py` : Script that takes the protein
  distribution moments as inferred from the numerical integration of the
  dynamical equations and computes the corresponding Lagrange multipliers for a
  maximum entropy approximation of the distribution for a finer grid of inducer
  concentrations.
    
- `maxent_protein_noise_dist.py` : Script that updates the second and third
  moment of the protein distribution to match the factor of two in the
  deviation between the original theoretical prediction and the experimental
  data. It then uses these updated moments along with the first protein moment
  to infer the maximum entropy distribution.