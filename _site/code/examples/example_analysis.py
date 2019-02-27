"""
Title:
    example_analysis.py
Author:
    Griffin Chure and Manuel Razo-Mejia
Creation Date:
    20170220
Last Modified:
    20170220
Purpose:
    This script serves as a representative example of an analysis script. This
    reads in the csv file generated from the `example_processing.py` script and
    generates a set of plots that serve as quality control checks for the
    experiment.
"""

# Import dependencies
import os
import glob
import numpy as np
import pandas as pd
import scipy

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

# Set the plotting style.
import mwc_induction_utils as mwc
mwc.set_plotting_style()


# Define variables to use over the script
date = 20160825
username = 'mrazomej'
run = 'r2'
operator = 'O1'

# Read the CSV file with the mean fold change
df = pd.read_csv('example_' + str(date) + '_' + run + '_' + operator +
                 '_IPTG_titration_MACSQuant.csv', comment='#')
rbs = df.rbs.unique()

# Plot all raw data
plt.figure()
for strain in rbs[np.array([r != 'auto' and r != 'delta' for r in rbs])]:
    plt.plot(df[df.rbs == strain].sort_values(by='IPTG_uM').IPTG_uM * 1E-6,
             df[df.rbs == strain].sort_values(by='IPTG_uM').fold_change_A,
             marker='o', linewidth=1, linestyle='--', label=strain)
plt.xscale('log')
plt.xlabel('IPTG (M)')
plt.ylabel('fold-change')
plt.ylim([-0.01, 1.2])
plt.xlim([1E-8, 1E-2])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('example_' + operator + '_IPTG_titration_data.png')

# Plot the curve for the 0 IPTG cultures
repressor_array = np.logspace(0, 3, 200)
binding_energy = df.binding_energy.unique()
fc_theory = 1 / (1 + 2 * repressor_array / 5E6 * np.exp(- binding_energy))
plt.figure(figsize=(7, 7))
plt.plot(repressor_array, fc_theory)
no_iptg = df.groupby('IPTG_uM').get_group(0)
plt.plot(no_iptg.repressors, no_iptg.fold_change_A, marker='o', linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('repressor copy number')
plt.ylabel('fold-change')
plt.tight_layout()
plt.savefig('example_' + operator + '_lacI_titration_ctrl.png')
