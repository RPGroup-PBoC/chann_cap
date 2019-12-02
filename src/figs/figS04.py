#%%
import os
import pickle
import cloudpickle
import itertools
import glob
import numpy as np
import scipy as sp
import pandas as pd
import git

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
# Increase dpi

#%%

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Define directories for data and figure 
figdir = f'{homedir}/fig/si/'
datadir = f'{homedir}/data/mRNA_FISH/'

# %%

# Read the data
df = pd.read_csv(f'{datadir}Jones_Brewster_2014.csv', index_col=0)

# Extract the lacUV5 data
dfUV5 = df[df.experiment == 'UV5']

# Compute the area ECDF
x, y = ccutils.stats.ecdf(dfUV5['area_cells'])

# Find the value to separate small from large cells
frac = 1 / 3
fraction = 2 * (1 - 2**(-frac))
idx = (np.abs(y - fraction)).argmin()
threshold = x[idx]

# Define colors for each group of cells
colors = sns.color_palette('Blues', n_colors=3)[1::]

with sns.axes_style('white', {'axes.spines.bottom': False,
                              'axes.spines.left': False,
                              'axes.spines.right': False,
                              'axes.spines.top': False}):
    # Plot this ECDF
    plt.plot(x[::20], y[::20], lw=0, marker='.', color='k')
    # Plot vertical line next to the threshold
    plt.plot([threshold, threshold], [0, 1], color='black', linestyle='--')

    # Fill the area for small and large cells
    plt.axvspan(0, threshold, alpha=0.75, color=colors[0])
    plt.axvspan(threshold, max(x[::20]), alpha=0.75, color=colors[1])

    # Label as small and large cells
    plt.text(100, 0.5, 'small cells', fontsize=10)
    plt.text(375, 0.5, 'large cells', fontsize=10)

    # Label plot
    plt.xlabel('area (pixels)')
    plt.ylabel('ecdf')
    plt.margins(0.025)

# Save figure
plt.tight_layout()
plt.savefig(figdir + 'figS04.pdf', bbox_inches='tight')
