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
# Keep only percentile required
percentile = 0.95
df_noise = df_noise[df_noise.percentile == percentile]

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
fig = plt.figure(figsize=(5, 3))
# Define outer grispec to keep at top the fold-change and at the bottom
# the noise
gs_out = mpl.gridspec.GridSpec(
    2, 1, height_ratios=[1, 1 + 1 / 5], hspace=0.1, wspace=0.05
)

# make nested gridspecs
gs_fc = mpl.gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_out[0], wspace=0.05
)
gs_noise = mpl.gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=gs_out[1],
    wspace=0.05,
    hspace=0.01,
    height_ratios=[1, 5],
)

# Add axis to plots
# fold-change
ax_fc = [plt.subplot(gs) for gs in gs_fc]
# noise
ax_noise = [plt.subplot(gs) for gs in gs_noise]

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Plot fold-change
    # Linear
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

    # Plot noise
    # Linear
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

# Define data threshold
dthresh = 5.9
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot fold_change
    ax_fc[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.fold_change,
        yerr=[
            data.fold_change - data.fold_change_lower,
            data.fold_change_upper - data.fold_change,
        ],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]] + 3].errorbar(
        x=data[data.noise <= dthresh].IPTG_uM,
        y=data[data.noise <= dthresh].noise,
        yerr=[data[data.noise <= dthresh].noise - 
        data[data.noise <= dthresh].noise_lower, 
        data[data.noise <= dthresh].noise_upper - 
        data[data.noise <= dthresh].noise],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        color="w",
        markeredgecolor=col_dict[group[0]][group[1]],
        label="",
        lw=0,
        marker="o",
        markersize=2,
    )

##  Set shared axis

# fold-change
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_fc[i]
    # join axis with first plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_fc[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Remove x ticks from left plot
plt.setp(ax_fc[0].get_xticklabels(), visible=False)
# Set axis to be shared with left lower plot
ax_fc[0].get_shared_x_axes().join(ax_fc[0], ax_noise[3])

# noise upper
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_noise[i]
    # join x axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    # join y axis with upper left plot
    ax.get_shared_y_axes().join(ax, ax_noise[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Set upper left plot x axis to be shared with lower left plot
ax.get_shared_x_axes().join(ax_noise[0], ax_noise[3])
# Remove x ticks from left plot
plt.setp(ax_noise[0].get_xticklabels(), visible=False)

# noise lower
# Loop through axis
for i in range(4, 6):
    # Select axis
    ax = ax_noise[i]
    # join axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_noise[3])
    # Remove y ticks labels
    plt.setp(ax.get_yticklabels(), visible=False)

# Set scales of reference plots and the other ones will follow
ax_noise[3].set_xscale("symlog", linthreshx=thresh)  # , linscalex=0.5)
ax_noise[0].set_yscale("log")

# Set limits
for i in range(3):
    ax_fc[i].set_ylim([-0.05, 1.4])

ax_noise[0].set_ylim([dthresh, 5e2])
ax_noise[3].set_ylim([-0.25, dthresh])

# Label axis
for i, ax in enumerate(ax_fc):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax.set_title(label, bbox=dict(facecolor="#ffedce"))
    # Set legend
    leg = ax.legend(title="rep./cell", fontsize=5)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=5)
    leg2 = ax_noise[i + 3].legend(
        title="rep./cell", fontsize=5, loc="upper right"
    )
    plt.setp(leg2.get_title(), fontsize=5)

    ax_noise[i + 3].set_xlabel(r"IPTG ($\mu$M)")

# Set ticks for the upper noise plot
ax_noise[0].set_yticks([1e1, 1e2])
ax_noise[1].set_yticks([1e1, 1e2])
ax_noise[2].set_yticks([1e1, 1e2])

# Add y axis labels
ax_fc[0].set_ylabel(r"fold-change")
ax_noise[3].set_ylabel(r"noise")

# Align y axis labels
fig.align_ylabels()

plt.savefig(figdir + "fig03C_v2.pdf", bbox_inches="tight")
# plt.savefig(figdir + "fig03C_v2.svg", bbox_inches="tight")
# plt.savefig(figdir + "fig03C_v2.png", bbox_inches="tight")
