#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import re
import git

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Import the project utils
import ccutils

# Set PBoC plotting format
ccutils.viz.set_plotting_style()

#%%

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Define directories for data and figure 
figdir = f'{homedir}/fig/si/'

df_cc_single = pd.read_csv(
    f"{homedir}/data/csv_maxEnt_dist/chann_cap_single_prom_protein.csv"
)

# Drop infinities
df_cc_single = df_cc_single[df_cc_single.channcap != np.inf]

# Read channel capacity of multi promoter model
df_cc_protein = pd.read_csv(f'{homedir}/data/csv_maxEnt_dist/' + 
                            'chann_cap_multi_prom_protein.csv')
# Drop infinities
df_cc_protein = df_cc_protein[df_cc_protein.channcap != np.inf]

# Group data by operator
df_group = df_cc_protein.groupby('operator')

# Define colors for each operator
operators = df_cc_protein['operator'].unique()
colors = sns.color_palette('colorblind', n_colors=len(operators))
op_col_dict = dict(zip(operators, colors))
op_dict = dict(zip(df_cc_protein.operator.unique(),
                   df_cc_protein.binding_energy.unique()))

# Define threshold for log vs linear section
thresh = 1E0

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize=(3.5,2.8))

# Plot multi-promoter data
for group, data in df_group:
    # Select x and y data for smoothing
    x = np.log10(data[data.repressor >= thresh].repressor.values)
    y = data[data.repressor >= thresh].channcap.values
    # Define lambda parameter for smoothing
    lam = 0.21
    # Smooth the channel capacity
    channcap_gauss = ccutils.stats.nw_kernel_smooth(x, x, y,lam)
    # Plot Log scale
    ax.plot(data[data.repressor >= thresh].repressor,
               channcap_gauss, 
               label=op_dict[group], color=op_col_dict[group])
    
#  # Group data by operator
df_group = df_cc_single.groupby('operator')

# Plot single-promoter
for group, data in df_group:
    # Select x and y data for smoothing
    x = np.log10(data[data.repressor >= thresh].repressor.values)
    y = data[data.repressor >= thresh].channcap.values
    # Define lambda parameter for smoothing
    lam = 0.21
    # Smooth the channel capacity
    channcap_gauss = ccutils.stats.nw_kernel_smooth(x, x, y,lam)
    # Plot Log scale
    ax.plot(data[data.repressor >= thresh].repressor,
            channcap_gauss, 
            label=op_dict[group], color=op_col_dict[group],
            linestyle='-.')
    
# Add artificial plots to add legend
ax.plot([], [], linestyle='-.', color='k', label='single-promoter')
ax.plot([], [], linestyle='-', color='k', label='multi-promoter')
    
# Increase y limit

# Label plot
ax.set_xlabel('repressor copy number')
ax.set_ylabel('channel capacity (bits)')
ax.set_xscale('log')
ax.legend(loc='upper left', title=r'$\Delta\epsilon_r \; (k_BT)$',
          bbox_to_anchor=(1, 0.75))
    
plt.savefig(figdir + "figS28.pdf", bbox_inches="tight")

