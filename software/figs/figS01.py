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

# Plot the histogram of the data with bins of width 1
_ = plt.hist(dfUV5.mRNA_cell, bins=np.arange(0, dfUV5.mRNA_cell.max()),
             density=1, histtype='stepfilled', align='left', lw=0)

# Label the plot
plt.xlabel('mRNA / cell')
plt.ylabel('probability')
plt.tight_layout()

plt.savefig(figdir + 'figS01.pdf', bbox_inches='tight')
