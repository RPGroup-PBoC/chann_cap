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
x, y = ccutils.stats.ecdf(dfUV5["area_cells"])

# Initialize array to save size classification
size = np.empty(len(dfUV5), dtype=str)

# Define threshold for small cells
frac = (1 / 3) - (1 / 10)
fraction = 2 * (1 - 2 ** (-frac))
idx = (np.abs(y - fraction)).argmin()
threshold = x[idx]
# Determine which cells are considered small
size[dfUV5.area_cells < threshold] = "s"

# Define threshold for large cells
frac = (1 / 3) + (1 / 10)
fraction = 2 * (1 - 2 ** (-frac))
idx = (np.abs(y - fraction)).argmin()
threshold = x[idx]
# Determine which cells are considered large
size[dfUV5.area_cells >= threshold] = "l"

# Save information on data frame
dfUV5 = dfUV5.assign(size=size)

# Remove unassigned cells
dfUV5 = dfUV5[dfUV5["size"] != ""]

# Group them by size
df_group = dfUV5.groupby("size")

# Define labels for cell size
exps = ["large", "small"]

# indicate bins
x = np.arange(dfUV5["mRNA_cell"].max() + 1)

# Initialize array to save distributions
px = np.zeros([len(exps), len(x)])

# Loop through each group and save probability
for i, (group, data) in enumerate(df_group):
    prob = data.mRNA_cell.value_counts(normalize=True, sort=False)
    px[i, prob.index] = prob.values

# Define colors
cmap = sns.color_palette("Blues", n_colors=3)[::-1]

ccutils.viz.pmf_cdf_plot(
    x,
    px,
    exps,
    xlabel="mRNA / cell",
    marker_size=100,
    color_palette=cmap,
    pmf_alpha=0.3,
    pmf_edgecolor=cmap,
    ylim=[0, np.max(px) * 1.2],
    cbar_label="cell size",
)

# Save figure
plt.savefig(figdir + 'figS05.pdf', bbox_inches='tight')
