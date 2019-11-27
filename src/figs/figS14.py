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

# Read moments for multi-promoter model
df_mom_rep = pd.read_csv(datadir + 'MaxEnt_multi_prom_constraints.csv')

# Read experimental determination of noise
df_noise = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m1p0)
mean_p_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m0p1)

# Compute the noise for the multi-promoter data
df_mom_rep = df_mom_rep.assign(
    m_noise=(
        np.sqrt(df_mom_rep.m2p0 - df_mom_rep.m1p0 ** 2) / df_mom_rep.m1p0
    ),
    p_noise=(
        np.sqrt(df_mom_rep.m0p2 - df_mom_rep.m0p1 ** 2) / df_mom_rep.m0p1
    ),
    m_fold_change=df_mom_rep.m1p0 / mean_m_delta,
    p_fold_change=df_mom_rep.m0p1 / mean_p_delta,
)

# Initialize list to save theoretical noise
thry_noise = list()
# Iterate through rows
for idx, row in df_noise.iterrows():
    # Extract information
    rep = float(row.repressor)
    op = row.operator
    if np.isnan(row.IPTG_uM):
        iptg = 0
    else:
        iptg = row.IPTG_uM
    
    # Extract equivalent theoretical prediction
    thry = df_mom_rep[(df_mom_rep.repressor == rep) &
                       (df_mom_rep.operator == op) &
                       (df_mom_rep.inducer_uM == iptg)].p_noise
    # Append to list
    thry_noise.append(thry.iloc[0])
    
df_noise = df_noise.assign(noise_theory = thry_noise)

#%%
# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(5, 2))

# Linear scale

# Plot reference line
ax[0].plot([1e-2, 1e2], [1e-2, 1e2], "--", color="gray")

# Plot error bars
ax[0].errorbar(
    x=df_noise.noise_theory,
    y=df_noise.noise,
    yerr=[
        df_noise.noise - df_noise.noise_lower,
        df_noise.noise_upper - df_noise.noise,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

# Plot data with color depending on log fold-change
ax[0].scatter(
    df_noise.noise_theory,
    df_noise.noise,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[0].set_xlabel("theoretical noise")
ax[0].set_ylabel("experimental noise")
ax[0].set_title("linear scale")

ax[0].set_xlim(0, 4)
ax[0].set_ylim(0, 4)
ax[0].set_xticks([0, 1, 2, 3, 4])
ax[0].set_yticks([0, 1, 2, 3, 4])

# Log scale

# Plot reference line
line = [1e-1, 1e2]
ax[1].loglog(line, line, "--", color="gray")
# Plot data with color depending on log fold-change

ax[1].errorbar(
    x=df_noise.noise_theory,
    y=df_noise.noise,
    yerr=[
        df_noise.noise - df_noise.noise_lower,
        df_noise.noise_upper - df_noise.noise,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

plot = ax[1].scatter(
    df_noise.noise_theory,
    df_noise.noise,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[1].set_xlabel("theoretical noise")
ax[1].set_ylabel("experimental noise")
ax[1].set_title("log scale")
ax[1].set_xlim([0.1, 10])

# show color scale
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot, cax=cbar_ax, ticks=[0, -1, -2, -3])

cbar.ax.set_ylabel("fold-change")
cbar.ax.set_yticklabels(["1", "0.1", "0.01", "0.001"])
cbar.ax.tick_params(width=0)

plt.subplots_adjust(wspace=0.4)
plt.savefig(figdir + "figS14.pdf", bbox_inches="tight")
