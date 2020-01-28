#%%
import os
import pickle
import cloudpickle
import itertools
import glob
import numpy as np
import scipy.special
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
mcmcdir = f'{homedir}/data/mcmc/'
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

dfUV5_small = dfUV5[dfUV5["area_cells"] < threshold]
dfUV5_large = dfUV5[dfUV5["area_cells"] > threshold]

# Splot DataFrame by area
dfUV5_large = dfUV5[dfUV5["area_cells"] > threshold]

# Load the flat-chain
with open(
    f"{mcmcdir}lacUV5_constitutive_mRNA_double_expo.pkl", "rb"
) as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# Generate a Pandas Data Frame with the mcmc chain
index = ["kp_on", "kp_off", "rm"]

# Generate a data frame out of the MCMC chains
df_mcmc = pd.DataFrame(gauss_flatchain, columns=index)

# rerbsine the index with the new entries
index = df_mcmc.columns

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
kpon_double, kpoff_double, rm_double = df_mcmc.iloc[max_idx, :]

# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), sharey=False, sharex=True)

##  Plot the single distribution  ##
# Define bins
bins = np.arange(0, dfUV5.mRNA_cell.max())

# Compute the probability using a two-copy promoter
frac = 1 / 3
fraction = 2 * (1 - 2 ** (-frac))

logp_mRNA_double = fraction * ccutils.model.log_p_m_unreg(
    bins, kpon_double, kpoff_double, 1, rm_double
) + (1 - fraction) * ccutils.model.log_p_m_unreg(
    bins, kpon_double, kpoff_double, 1, 2 * rm_double
)
# Re-Normalize distribution
logp_mRNA_double = logp_mRNA_double - scipy.special.logsumexp(logp_mRNA_double)

# Plot the histogram of the data with bins of width 1
_ = ax[0].hist(
    dfUV5.mRNA_cell,
    bins=bins,
    density=1,
    histtype="stepfilled",
    alpha=0.75,
    label="sm-FISH data",
    align="left",
    lw=0,
)

ax[0].step(bins, np.exp(logp_mRNA_double), lw=1.5, label="multi-promoter fit")

##  Plot split distributions  ##
# Define colors for each group of cells
colors = sns.color_palette("Blues", n_colors=3)[1::]

# Compute the probability
logp_mRNA_small = ccutils.model.log_p_m_unreg(
        bins, kpon_double, kpoff_double, 1, rm_double
)
logp_mRNA_large = ccutils.model.log_p_m_unreg(
    bins, kpon_double, kpoff_double, 1, 2 * rm_double
)

# Plot the histogram of the data with bins of width 1
ax[1].hist(
    dfUV5_small.mRNA_cell,
    bins=bins,
    density=1,
    histtype="stepfilled",
    alpha=0.3,
    label="small cells sm-FISH",
    align="left",
    color=colors[0],
    edgecolor=colors[0],
)
ax[1].hist(
    dfUV5_small.mRNA_cell,
    bins=bins,
    density=1,
    histtype="step",
    label="",
    align="left",
    lw=0.5,
    edgecolor=colors[0],
)


ax[1].hist(
    dfUV5_large.mRNA_cell,
    bins=bins,
    density=1,
    histtype="stepfilled",
    alpha=0.3,
    label="large cells sm-FISH",
    align="left",
    color=colors[1],
    edgecolor=colors[1],
    lw=2,
)
ax[1].hist(
    dfUV5_large.mRNA_cell,
    bins=bins,
    density=1,
    histtype="step",
    label="",
    align="left",
    lw=0.5,
    edgecolor=colors[1],
)

# Plot theoretical predictions

ax[1].step(
    bins,
    np.exp(logp_mRNA_small),
    color=colors[0],
    ls="-",
    lw=1.5,
    label="one promoter",
)
ax[1].step(
    bins,
    np.exp(logp_mRNA_large),
    color=colors[1],
    ls="-",
    lw=1.5,
    label="two promoters",
)

# Label the plots
ax[0].set_xlabel("mRNA / cell")
ax[1].set_xlabel("mRNA / cell")

ax[0].set_ylabel("probability")
ax[1].set_ylabel("probability")

# Set legend
ax[0].legend()
ax[1].legend()

# Add labels to plots
plt.figtext(0.01, 0.9, "(A)", fontsize=8)
plt.figtext(0.5, 0.9, "(B)", fontsize=8)

# Save figure
plt.tight_layout()
plt.savefig(f'{figdir}/figS07.pdf', bbox_inches='tight')
