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
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(datadir + "MaxEnt_Lagrange_mult_protein.csv")

# Extract protein moments in constraints
prot_mom = [x for x in df_maxEnt.columns if "lambda_m0" in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r"\d+", s))) for s in prot_mom]

# Define operators to be included
operators = ["O1", "O2", "O3"]

# Define repressors to be included
repressors = [22, 260, 1740]

# Define color for operators
# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
col_dict = dict(zip(operators, col_list))

# Define binstep for plot, meaning how often to plot
# an entry
binstep = 100

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 1.3e4)

# Initialize plot
fig, ax = plt.subplots(
    len(repressors), len(operators), figsize=(5, 5), sharex=True, sharey=True
)

# Loop through operators
for j, op in enumerate(operators):
    # Loop through repressors
    for i, rep in enumerate(repressors):
        # Extract the multipliers for a specific strain
        df_sample = df_maxEnt[
            (df_maxEnt.operator == op) & (df_maxEnt.repressor == rep)
        ]

        # Group multipliers by inducer concentration
        df_group = df_sample.groupby("inducer_uM")

        # Define colors for plot
        colors = sns.color_palette(col_dict[op], n_colors=len(df_group) + 1)

        # Initialize matrix to save probability distributions
        Pp = np.zeros([len(df_group), len(protein_space)])

        # Loop through each of the entries
        for k, (group, data) in enumerate(df_group):
            # Select the Lagrange multipliers
            lagrange_sample = data.loc[
                :, [col for col in data.columns if "lambda" in col]
            ].values[0]

            # Compute distribution from Lagrange multipliers values
            Pp[k, :] = ccutils.maxent.maxEnt_from_lagrange(
                mRNA_space, protein_space, lagrange_sample, exponents=moments
            ).T

            # CDF plot
            ax[i, j].plot(
                protein_space[0::binstep],
                np.cumsum(Pp[k, :])[0::binstep],
                drawstyle="steps",
                color=colors[k],
                linewidth=2,
            )

        # Add x label to lower plots
        if i == 2:
            ax[i, j].set_xlabel("protein / cell")

        # Add y label to left plots
        if j == 0:
            ax[i, j].set_ylabel("CDF")

        # Add operator top of colums
        if i == 0:
            label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(
                df_sample.binding_energy.unique()[0]
            )
            ax[i, j].set_title(label, bbox=dict(facecolor="#ffedce"))

        # Add repressor copy number to right plots
        if j == 2:
            # Generate twin axis
            axtwin = ax[i, j].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(rep),
                bbox=dict(facecolor="#ffedce"),
            )
            # Remove residual ticks from the original left axis
            ax[i, j].tick_params(color="w", width=0)

# Adjust spacing between plots
plt.subplots_adjust(hspace=0.02, wspace=0.02)

plt.savefig(figdir + "figS16_v3.pdf", bbox_inches="tight")