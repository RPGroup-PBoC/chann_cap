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
figdir = f'{homedir}/fig/si/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'

# %%
#%%
# Read moments for multi-promoter model
df_mom_iptg = pd.read_csv(datadir + 'MaxEnt_multi_prom_IPTG_range.csv')

# Compute the skewness for the multi-promoter data
m_mean = df_mom_iptg.m1p0
p_mean = df_mom_iptg.m0p1
m_var = df_mom_iptg.m2p0 - df_mom_iptg.m1p0 ** 2
p_var = df_mom_iptg.m0p2 - df_mom_iptg.m0p1 ** 2

df_mom_iptg = df_mom_iptg.assign(
    m_skew=(df_mom_iptg.m3p0 - 3 * m_mean * m_var - m_mean**3)
    / m_var**(3 / 2) * 2,
    p_skew=(df_mom_iptg.m0p3 - 3 * p_mean * p_var - p_mean**3)
    / p_var**(3 / 2) * 2,
)

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

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
energies = [-15.3, -13.9, -9.7]
op_idx = dict(zip(operators, np.arange(3)))

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
# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_skew,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # linear scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_skew,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    ax[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.skewness,
        yerr=[data.skewness - data.skewness_lower, 
        data.skewness_upper - data.skewness],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )


for i, a in enumerate(ax):
    # systematically change axis for all subplots
    ax[i].set_xscale("symlog", linthreshx=thresh, linscalex=0.5)
    # Set legend
    leg = ax[i].legend(title="rep./cell", fontsize=8)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)

    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i].set_xlabel(r"IPTG (µM)")
ax[0].set_ylabel(r"skewness")

# Change spacing between plots
plt.subplots_adjust(wspace=0.05)
plt.savefig(figdir + "figS18.pdf", bbox_inches="tight")
