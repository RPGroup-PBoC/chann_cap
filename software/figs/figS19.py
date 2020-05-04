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
operators = ["O1", "O2", "O3"]
energies = [-15.3, -13.9, -9.7]

# Define repressor to be included
repressor = [22, 260, 1740]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define color for operators
# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
col_dict = dict(zip(operators, col_list))

# Initialize figure
fig = plt.figure(figsize=(7 * 0.8, 15 * 0.6))
# Define outer grid
outer = mpl.gridspec.GridSpec(len(operators), 1, hspace=0.3)

# Loop through operators
for k, op in enumerate(operators):
    # Initialize inner grid
    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        len(rep), len(inducer), subplot_spec=outer[k], wspace=0.02, hspace=0.05
    )

    # Define colors
    colors = sns.color_palette(col_dict[op], n_colors=len(inducer) + 2)

    # Loop through repressor copy numbers
    for j, r in enumerate(rep):
        # Loop through concentrations
        for i, c in enumerate(inducer):
            # Initialize subplots
            ax = plt.Subplot(fig, inner[j, i])

            # Add subplot to figure
            fig.add_subplot(ax)

            # Extract data
            data = df_micro[
                (df_micro.repressor == r)
                & (df_micro.operator == op)
                & (df_micro.IPTG_uM == c)
            ]

            # generate experimental ECDF
            x, y = ccutils.stats.ecdf(data.fold_change)

            # Plot ECDF
            ax.plot(
                x[::binstep],
                y[::binstep],
                color=colors[i],
                alpha=1,
                lw=2,
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
            ax.plot(
                fc_space[0::binstep_theory],
                np.cumsum(Pp)[0::binstep_theory],
                linestyle="--",
                color="k",
                alpha=0.75,
                linewidth=1,
                label="",
            )

            # Label x axis
            if j == len(rep) - 1:
                ax.set_xlabel("fold-change", fontsize=7.5)

            # Label y axis
            if i == 0:
                ax.set_ylabel("ECDF")

            # Add title to plot
            if j == 0:
                ax.set_title(
                    r"{:.0f} ($\mu M$)".format(c),
                    color="white",
                    bbox=dict(facecolor=colors[i]),
                    fontsize=7
                )

            # Remove x ticks and y ticks from middle plots
            if i != 0:
                ax.set_yticklabels([])
            if j != len(rep) - 1:
                ax.set_xticklabels([])

            # Add repressor copy number to right plots
            if i == len(inducer) - 1:
                # Generate twin axis
                axtwin = ax.twinx()
                # Remove ticks
                axtwin.get_yaxis().set_ticks([])
                # Set label
                axtwin.set_ylabel(
                    r"rep. / cell = {:d}".format(r),
                    bbox=dict(facecolor="#ffedce"),
                    fontsize=5
                )
                # Remove residual ticks from the original left axis
                ax.tick_params(color="w", width=0)

            if (j == 1) and (i == len(inducer) - 1):
                text = ax.text(
                    1.35,
                    0.5,
           r"$\Delta\epsilon_r = {:.1f} \; k_BT$".format(energies[k]),
                    size=8,
                    verticalalignment="center",
                    rotation=90,
                    color="white",
                    transform=ax.transAxes,
                    bbox=dict(facecolor=colors[0]),
                )

plt.savefig(figdir + "figS19.pdf", bbox_inches="tight")
