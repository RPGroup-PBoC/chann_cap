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

df_micro = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       '20181003_O2_RBS1027_IPTG_titration_microscopy.csv',
                       comment='#')

# Extract the data from the experimental strain
df_exp = df_micro[df_micro.rbs == "RBS1027"]

# Set the number of bins and the fraction of data to use
fracs = np.linspace(0.1, 1, 10)
bins = np.floor(np.linspace(10, 100, 10)).astype(int)
nreps = 15

# Define function to perform the computation in paralel
def channcap_bs_parallel(b):
    # Initialize matrix to save bootstrap repeats
    MI_bs = np.zeros([len(fracs), nreps])
    samp_sizes = np.zeros(len(fracs))
    for i, frac in enumerate(fracs):
        MI_bs[i, :], samp_sizes[i] = ccutils.channcap.channcap_bootstrap(
            df_exp, bins=b, nrep=nreps, frac=frac
        )
    return (MI_bs, samp_sizes)

channcap_list = Parallel(n_jobs=6)(
    delayed(channcap_bs_parallel)(b) for b in bins
)

# Define elements to extract from the microscopy data frame to add to the
# Bootstrap data frame
kwarg_list = [
    "date",
    "username",
    "operator",
    "binding_energy",
    "rbs",
    "repressors",
]
kwargs = dict((x, df_exp[x].unique()[0]) for x in kwarg_list)
df_cc_bs = ccutils.channcap.tidy_df_channcap_bs(channcap_list, fracs, bins, **kwargs)

# Group by the number of bins
df_group = df_cc_bs.groupby("bins")
# Initialize arrays to save the slope and intercept of the linear regression
lin_reg = np.zeros([len(bins), 2])

# Loop through each bin size and find the intercept
for i, (group, data) in enumerate(df_group):
    # Define the inverse sample size as x values
    x = 1 / data.samp_size
    # Set channel capacity as y values
    y = data.channcap_bs
    # Perform the linear regression
    lin_reg[i, :] = np.polyfit(x, y, deg=1)

# Plot
df_bin_group = df_cc_bs.groupby(["bins", "samp_size"])

bins = df_cc_bs.bins.unique()
bin_color = dict(zip(bins, sns.color_palette("viridis_r", n_colors=len(bins))))

fig, ax = plt.subplots(1, 1)
# Define the xlims that will use as evaluating points for the linear regression
xlims = [0, 2e-3]
# add legend and line
for i, b in enumerate(bins):
    ax.errorbar([], [], color=bin_color[b], label=b, fmt="o")
    ax.plot(
        xlims, np.polyval(lin_reg[i, :], xlims), color=bin_color[b], label=None
    )

for group, data in df_bin_group:
    ax.errorbar(
        x=1 / group[1],
        y=data["channcap_bs"].mean(),
        yerr=data["channcap_bs"].std(),
        fmt="o",
        color=bin_color[group[0]],
        label=None,
        markersize=3,
    )

ax.legend(loc="center left", title="# bins", bbox_to_anchor=(1.0, 0.5))

# Set limits
ax.set_xlim(left=0)
ax.set_ylim(bottom=1)

# Label axis
ax.set_xlabel(r"(sample size)$^{-1}$")
ax.set_ylabel(r"$I_{biased}$ (bits)")

# Set a nice scientific notation for the x axis
ax.ticklabel_format(axis="x",
                    style="sci",
                    scilimits=(0, 0),
                    useMathText=True)


plt.savefig(figdir + "figS31.pdf", bbox_inches="tight")

