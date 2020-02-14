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
operators = ["O1", "O2", "O3"]

# Remove these dates
df_micro = pd.read_csv(
    "../../data/csv_microscopy/single_cell_microscopy_data.csv"
)

df_micro[["date", "operator", "rbs", "mean_intensity", "intensity"]].head()

# group df by date
df_group = df_micro.groupby("date")

# loop through dates
for group, data in df_group:
    # Extract mean autofluorescence
    mean_auto = data[data.rbs == "auto"].mean_intensity.mean()
    # Extract ∆lacI data
    delta = data[data.rbs == "delta"]
    mean_delta = (delta.intensity - delta.area * mean_auto).mean()
    # Compute fold-change
    fc = (data.intensity - data.area * mean_auto) / mean_delta
    # Add result to original dataframe
    df_micro.loc[fc.index, "fold_change"] = fc

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 2.2e4)

# Extract the multipliers for a specific strain
df_maxEnt_delta = df_maxEnt[
    (df_maxEnt.operator == "O1")
    & (df_maxEnt.repressor == 0)
    & (df_maxEnt.inducer_uM == 0)
]

# Select the Lagrange multipliers
lagrange_sample = df_maxEnt_delta.loc[
    :, [col for col in df_maxEnt_delta.columns if "lambda" in col]
].values[0]

# Compute distribution from Lagrange multipliers values
Pp = ccutils.maxent.maxEnt_from_lagrange(
    mRNA_space,
    protein_space,
    lagrange_sample,
    exponents=moments
).T

# Compute mean protein copy number
mean_delta_p = np.sum(protein_space * Pp)

# Transform protein_space into fold-change
fc_space = protein_space / mean_delta_p

##  Plot ECDF for experimental data
# Keep only data for ∆lacI
df_delta = df_micro[df_micro.rbs == "delta"]

# Group data by operator
df_group = df_delta.groupby("operator")

# Initialize figure
fig, ax = plt.subplots(3, 1, figsize=(5 / 3, 5), sharex=True, sharey=True)

# Define colors for operators
col_list = ["Blues_r", "Reds_r", "Greens_r"]
col_dict = dict(zip(("O1", "O2", "O3"), col_list))

# Loop through operators
for i, (group, data) in enumerate(df_group):
    # Group data by date
    data_group = data.groupby("date")
    # Generate list of colors
    colors = sns.color_palette(col_dict[group], n_colors=len(data_group))

    # Loop through dates
    for j, (g, d) in enumerate(data_group):
        # Generate ECDF
        x, y = ccutils.stats.ecdf(d.fold_change)
        # Plot ECDF
        ax[i].plot(
            x[::10],
            y[::10],
            lw=0,
            marker=".",
            color=colors[j],
            alpha=0.3,
            label="",
        )

    # Label y axis
    ax[i].set_ylabel("ECDF")
    # Generate twin axis
    axtwin = ax[i].twinx()
    # Remove ticks
    axtwin.get_yaxis().set_ticks([])
    # Set label
    label = r"operator {:s}".format(group)
    axtwin.set_ylabel(
        label,
        bbox=dict(facecolor="#ffedce"),
    )
    # Remove residual ticks from the original left axis
    ax[i].tick_params(color="w", width=0)
    # label = r"operator {:s}".format(group)
    # ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))

    # Plot theoretical prediction
    ax[i].plot(
        fc_space[0::100],
        np.cumsum(Pp)[0::100],
        linestyle="--",
        color="k",
        linewidth=1.5,
        label="theory",
    )

    
    # Add fake data point for legend
    ax[i].plot([], [], lw=0, marker=".", color=colors[0], label="microscopy")
    # Add legend
    # ax[i].legend()

# Label y axis of left plot
ax[2].set_xlabel("fold-change")

# Change limit
ax[0].set_xlim(right=3)

# Change spacing between plots
plt.subplots_adjust(hspace=0.02)

plt.savefig(figdir + "fig04B_v2.pdf", bbox_inches="tight")
