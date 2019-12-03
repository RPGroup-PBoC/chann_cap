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
# Define repressor copy number and operator
rep = [22, 260, 1740]
# Define binstep for plot
binstep = 10
binstep_theory = 100

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
fc_space = protein_space / mean_delta_p# Define operators to be included

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define repressor copy number and operator
rep = [22, 260, 1740]
op = "O2"

# Define binstep for plot
binstep = 10
binstep_theory = 100

# Define colors
colors = sns.color_palette("Oranges_r", n_colors=len(inducer) + 2)

# Initialize plot
fig, ax = plt.subplots(
    len(rep), len(inducer), figsize=(7, 4.5), sharex=True, sharey=True
)

# Loop through repressor copy numbers
for j, r in enumerate(rep):
    # Loop through concentrations
    for i, c in enumerate(inducer):
        # Extract data
        data = df_micro[
            (df_micro.repressor == r)
            & (df_micro.operator == op)
            & (df_micro.IPTG_uM == c)
        ]

        # generate experimental ECDF
        x, y = ccutils.stats.ecdf(data.fold_change)

        # Plot ECDF
        ax[j, i].plot(
            x[::binstep],
            y[::binstep],
            color=colors[i],
            alpha=1,
            lw=3,
            label="{:.0f}".format(c),
        )

        # Extract lagrange multiplieres
        df_me = df_maxEnt[
            (df_maxEnt.operator == op)
            & (df_maxEnt.repressor == r)
            & (df_maxEnt.inducer_uM == c)
        ]

        lagrange_sample = df_me.loc[
            :, [col for col in df_me.columns if "lambda" in col]
        ].values[0]

        # Compute distribution from Lagrange multipliers values
        Pp = ccutils.maxent.maxEnt_from_lagrange(
            mRNA_space, protein_space, lagrange_sample, exponents=moments
        ).T

        # Plot theoretical prediction
        ax[j, i].plot(
            fc_space[0::binstep_theory],
            np.cumsum(Pp)[0::binstep_theory],
            linestyle="--",
            color="k",
            alpha=0.75,
            linewidth=1.5,
            label="",
        )

        # Label x axis
        if j == len(rep) - 1:
            ax[j, i].set_xlabel("fold-change")

        # Label y axis
        if i == 0:
            ax[j, i].set_ylabel("ECDF")

        # Add title to plot
        if j == 0:
            ax[j, i].set_title(
                r"{:.0f} ($\mu M$)".format(c),
                color="white",
                bbox=dict(facecolor=colors[i]),
            )

        # Add repressor copy number to right plots
        if i == len(inducer) - 1:
            # Generate twin axis
            axtwin = ax[j, i].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(r), bbox=dict(facecolor="#ffedce")
            )
            # Remove residual ticks from the original left axis
            ax[j, i].tick_params(color="w", width=0)

fig.suptitle(
    r"$\Delta\epsilon_r = {:.1f}\; k_BT$".format(-13.9),
    bbox=dict(facecolor="#ffedce"),
    size=10,
)
plt.subplots_adjust(hspace=0.05, wspace=0.02)
plt.savefig(figdir + "figS23B.pdf", bbox_inches="tight")
