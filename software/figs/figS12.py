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
df_mom_iptg = pd.read_csv(datadir + 'MaxEnt_multi_prom_IPTG_range.csv')

# Read constraints for the single promoter model
df_mom_single = pd.read_csv(datadir + 'single_prom_moments.csv')

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

#%%
# Extract theoretical noise for the ∆lacI strain
noise_delta_thry = df_mom_iptg[df_mom_iptg.repressor == 0].p_noise.mean()
noise_delta_thry_single = df_mom_single[
    df_mom_single.repressor == 0
].p_noise.mean()

# Extract data with 95% percentile
df_delta = df_noise[(df_noise.repressor == 0) & (df_noise.percentile == 0.95)]

# Define colors for operators
col_list = ["Blues_r", "Reds_r", "Greens_r"]
colors = [sns.color_palette(x, n_colors=1) for x in col_list]

# Plot theoretical prediction

# Generate stripplot for experimentally determined
# noise of the ∆lacI strain
fig, ax = plt.subplots(1, 1)
ccutils.viz.jitterplot_errorbar(ax, df_delta, jitter=0.1)

# Plot theoretical prediction as a horizontal black line
ax.axhline(
    noise_delta_thry_single,
    color="gray",
    linestyle=":",
    label="single-promoter",
)
ax.axhline(noise_delta_thry, color="k", linestyle="--", label="multi-promoter")

# Include legend
ax.legend(title="model", loc="upper center")

# Set axis limits
ax.set_ylim([0, 1])

# Label axis
ax.set_ylabel(r"noise")

# Save figure
plt.tight_layout()
plt.savefig(figdir + "figS12.pdf", bbox_inches="tight")