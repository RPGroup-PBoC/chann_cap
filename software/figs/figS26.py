#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import re
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

#%%

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

# Define directories for data and figure 
figdir = f'{homedir}/fig/si/'
datadir = f'{homedir}/data/csv_gillespie/'

# %%
# Read simulations into memory
df_sim_mRNA = pd.read_csv(datadir + "two_state_mRNA_gillespie.csv")

# Extract data from last cell cycle
df = df_sim_mRNA[df_sim_mRNA.cycle == df_sim_mRNA.cycle.max()]

# Determine bins for histogram
bins = np.arange(df.mRNA.max())

# Import parameters
param = ccutils.model.load_constants()

kp_on = param['kp_on']
kp_off = param['kp_off']
rm = param['rm']
gm = param['gm']

# Compute the probability
logp_mRNA_small = ccutils.model.log_p_m_unreg(bins, kp_on, kp_off, gm, rm)
logp_mRNA_large = ccutils.model.log_p_m_unreg(bins, kp_on, kp_off, gm, 2 * rm)

# Group by promoter state
df_group = df.groupby("state")

# Initialize figure
fig = plt.figure()

# Define colors for each group of cells
colors = np.flip(sns.color_palette("Blues", n_colors=3)[1::], axis=0)

# Loop through states
for i, (group, data) in enumerate(df_group):
    # Extract time
    time = np.sort(data.time.unique())
    # Extract mRNA data
    mRNA = data[data.time == time[-20]].mRNA
    # Histogram data
    plt.hist(
        mRNA,
        bins=bins,
        density=1,
        histtype="stepfilled",
        alpha=0.3,
        label="gillespie " + group,
        align="left",
        color=colors[i],
        edgecolor=colors[i],
    )
    plt.hist(
        mRNA,
        bins=bins,
        density=1,
        histtype="step",
        label="",
        align="left",
        lw=0.5,
        color=colors[i],
        edgecolor=colors[i],
    )

# Plot theoretical predictions
plt.step(
    bins,
    np.exp(logp_mRNA_small),
    color=colors[1],
    linestyle="-",
    lw=1.5,
    label="analytical single",
)
plt.step(
    bins,
    np.exp(logp_mRNA_large),
    color=colors[0],
    linestyle="-",
    lw=1.5,
    label="analytical double",
)

# Label the plot
plt.xlabel("mRNA / cell")
plt.ylabel("probability")
_ = plt.legend()
plt.tight_layout()

plt.savefig(figdir + "figS26.pdf", bbox_inches="tight")
