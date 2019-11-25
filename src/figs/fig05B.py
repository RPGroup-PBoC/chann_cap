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
figdir = f'{homedir}/fig/main/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(datadir + "MaxEnt_Lagrange_mult_protein.csv")

# Extract protein moments in constraints
prot_mom = [x for x in df_maxEnt.columns if "lambda_m0" in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r"\d+", s))) for s in prot_mom]

# Define operators to be included
operator = ["O3"]

# Define repressors to be included
repressors = [1740, 22]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define binstep for plot, meaning how often to plot
# an entry
binstep = 100

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 1.8e4)

# Initialize plot
fig, ax = plt.subplots(
    len(repressors), 1, figsize=(5 / 3, 2.5), sharex=True, sharey=True
)

# Define displacement
displacement = 5e-5

# Loop through repressors
for i, rep in enumerate(repressors):

    # Extract the multipliers for a specific strain
    df_sample = df_maxEnt[
        (df_maxEnt.operator == operator[0])
        & (df_maxEnt.repressor == rep)
        & (df_maxEnt.inducer_uM.isin(inducer))
    ]

    # Group multipliers by inducer concentration
    df_group = df_sample.groupby("inducer_uM", sort=True)

    # Extract and invert groups to start from higher to lower
    groups = np.flip([group for group, data in df_group])

    # Define colors for plot
    colors = sns.color_palette("Greens", n_colors=len(df_group) + 1)

    # Initialize matrix to save probability distributions
    Pp = np.zeros([len(df_group), len(protein_space)])

    # Loop through each of the entries
    for k, group in enumerate(groups):
        data = df_group.get_group(group)

        # Select the Lagrange multipliers
        lagrange_sample = data.loc[
            :, [col for col in data.columns if "lambda" in col]
        ].values[0]

        # Compute distribution from Lagrange multipliers values
        Pp[k, :] = ccutils.maxent.maxEnt_from_lagrange(
            mRNA_space, protein_space, lagrange_sample, exponents=moments
        ).T

        # Generate PMF plot
        ax[i].plot(
            protein_space[0::binstep],
            Pp[k, 0::binstep] + k * displacement,
            drawstyle="steps",
            lw=1,
            color="k",
            zorder=len(df_group) * 2 - (2 * k),
        )
        # Fill between each histogram
        ax[i].fill_between(
            protein_space[0::binstep],
            Pp[k, 0::binstep] + k * displacement,
            [displacement * k] * len(protein_space[0::binstep]),
            color=colors[k],
            alpha=1,
            step="pre",
            zorder=len(df_group) * 2 - (2 * k + 1),
        )

    # Add x label to lower plots
    if i == 1:
        ax[i].set_xlabel("protein / cell")

    ax[i].set_ylabel("[IPTG] ($\mu$M)")

    # Change x axis font size
    ax[i].tick_params(axis="y", labelsize=5.5)

# Change lim
ax[0].set_ylim([-3e-5, 5.5e-4 + len(df_group) * displacement])
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.1, wspace=0.04)

# Set y axis ticks
yticks = np.arange(len(df_group)) * displacement
yticklabels = [int(x) for x in groups]

ax[0].yaxis.set_ticks(yticks)
ax[0].yaxis.set_ticklabels(yticklabels)
ax[0].yaxis.set_ticklabels(yticklabels)

# Set x axis ticks
xticks = [0, 5e3, 1e4, 1.5e4]
ax[0].xaxis.set_ticks(xticks)

plt.savefig(figdir + "fig05B.pdf", bbox_inches="tight")
