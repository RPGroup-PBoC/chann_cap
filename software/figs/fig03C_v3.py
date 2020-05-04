#%%
import os
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
figdir = f'{homedir}/fig/main/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%

# Read moments for multi-promoter model
df_mom_iptg = pd.read_csv(datadir + 'MaxEnt_multi_prom_IPTG_range.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m1p0
)
mean_p_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m0p1
)

# Compute the noise for the multi-promoter data
df_mom_iptg = df_mom_iptg.assign(
    m_noise=np.sqrt(df_mom_iptg.m2p0 - df_mom_iptg.m1p0**2) / 
            df_mom_iptg.m1p0,
    p_noise=np.sqrt(df_mom_iptg.m0p2 - df_mom_iptg.m0p1**2) / 
            df_mom_iptg.m0p1,
    m_fold_change=df_mom_iptg.m1p0 / mean_m_delta,
    p_fold_change=df_mom_iptg.m0p1 / mean_p_delta
)

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

df_noise = df_noise[df_noise.percentile == 0.95]
#%%
# Extract regulated promoter information
df_noise_reg = df_noise[df_noise.repressor > 0]
# Define repressor copy numbers to include
rep = df_noise_reg["repressor"].unique()

# Group moments by operator and repressor
df_group_exp = (
    df_noise_reg[df_noise_reg.noise > 0]
    .sort_values("IPTG_uM")
    .groupby(["operator", "repressor"])
)

df_group = (
    df_mom_iptg[df_mom_iptg["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

# Generate index for each opeartor
operators = ["O1", "O2", "O3"]
op_idx = dict(zip(operators, np.arange(3)))

# List energies
energies = [-15.3, -13.9, -9.7]

# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
# Loop through operators generating dictionary of colors for each
col_dict = {}
for i, op in enumerate(operators):
    col_dict[op] = dict(
        zip(rep, sns.color_palette(col_list[i], n_colors=len(rep) + 1)[0:3])
    )

# Define threshold to separate log scale from linear scale
thresh = 1e-1

#%%
# Initialize figure
fig, ax = plt.subplots(2, 3, figsize=(5, 3), sharex=True, sharey="row")

# Linearize plot numeration
ax = ax.ravel()

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Plot fold-change
    # Log scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # Linear scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

    # Plot noise
    # Log scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # Linear scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot fold_change
    ax[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.fold_change,
        yerr=[
            data.fold_change - data.fold_change_lower,
            data.fold_change_upper - data.fold_change,
        ],
        fmt="o",
        ms=2.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=2.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )


for i, a in enumerate(ax):
    # systematically change axis for all subplots
    ax[i].set_xscale("symlog", linthreshx=thresh, linscalex=1)
    # Set specifics for fold-change plots
    if i < 3:
        # Set title
        label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
        ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
        # Set legend on fold-change plot
        leg = ax[i].legend(title="rep./cell", fontsize=5.3)
        # Set legend font size
        plt.setp(leg.get_title(), fontsize=5.3)
        # Set legend on noise plot
        leg2 = ax[i + 3].legend(title="rep./cell", fontsize=5.3)
        # Set legend font size
        plt.setp(leg2.get_title(), fontsize=5.3)

    # Set specifics for noise plots
    else:
        ax[i].set_yscale("log")
        ax[i].set_ylim(bottom=1e-1)
        # Label axis
        ax[i].set_xlabel(r"IPTG ($\mu$M)")

ax[0].set_ylabel(r"fold-change")
ax[3].set_ylabel(r"noise")

# Change spacing between plots
plt.subplots_adjust(wspace=0.01, hspace=0.1)

plt.savefig(figdir + "fig03C_v3.pdf", bbox_inches="tight")