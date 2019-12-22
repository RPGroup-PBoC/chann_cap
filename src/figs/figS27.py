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
datadir = f'{homedir}/data/csv_gillespie/'

# %%
df_sim_prot = pd.read_csv(datadir + "two_state_protein_gillespie.csv")

# Extract mRNA data
mRNA_names = [x for x in df_sim_prot.columns if re.match(r"[m]\d", x)]
mRNA_data = df_sim_prot.loc[:, mRNA_names].values
# Compute mean mRNA
mRNA_mean = mRNA_data.mean(axis=1)

# Extract protein data
protein_names = [x for x in df_sim_prot.columns if re.match(r"[p]\d", x)]
protein_data = df_sim_prot.loc[:, protein_names].values
# Compute mean protein
protein_mean = protein_data.mean(axis=1)

# Initialize plot
fig, ax = plt.subplots(2, 1, figsize=(2.5, 2), sharex=True)

# Define colors
colors = sns.color_palette("Paired", n_colors=2)

# Define time stepsize for plot
binstep = 10
# Define every how many trajectories to plot
simnum = 10


# Plot mRNA trajectories
ax[0].plot(
    df_sim_prot["time"][0::binstep] / 60,
    mRNA_data[0::binstep, 0::simnum],
    color=colors[0],
)
# Plot mean mRNA
ax[0].plot(
    df_sim_prot["time"][0::binstep] / 60,
    mRNA_mean[0::binstep],
    color=colors[1],
)

# Plot protein trajectories
ax[1].plot(
    df_sim_prot["time"][0::binstep] / 60,
    protein_data[0::binstep, 0::simnum],
    color=colors[0],
)
# Plot mean protein
ax[1].plot(
    df_sim_prot["time"][0::binstep] / 60,
    protein_mean[0::binstep],
    color=colors[1],
)

# Group data frame by cell cycle
df_group = df_sim_prot.groupby("cycle")
# Loop through cycles
for i, (group, data) in enumerate(df_group):
    # Define the label only for the last cell cycle not to repeat in legend
    if group == df_sim_prot["cycle"].max():
        label_s = "single promoter"
        label_d = "two promoters"
    else:
        label_s = ""
        label_d = ""
    # Find index for one-promoter state
    idx = np.where(data.state == "single")[0]
    # Indicate states with two promoters
    ax[0].axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#e3dcd1",
        label=label_s,
    )
    ax[1].axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#e3dcd1",
        label=label_s,
    )

    # Find index for two-promoter state
    idx = np.where(data.state == "double")[0]
    # Indicate states with two promoters
    ax[0].axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#ffedce",
        label=label_d,
    )
    ax[1].axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#ffedce",
        label=label_d,
    )


# Set limits
ax[0].set_xlim(df_sim_prot["time"].min() / 60, df_sim_prot["time"].max() / 60)
# Label plot
ax[1].set_xlabel("time (min)")
ax[0].set_ylabel("mRNA/cell")
ax[1].set_ylabel("protein/cell")

# Set legend for both plots
ax[0].legend(
    loc="upper left",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(-0.12, 0, 0, 1.3),
    fontsize=6.5,
)

# Align y axis labels
fig.align_ylabels()

plt.subplots_adjust(hspace=0.05)

plt.savefig(figdir + "figS27.pdf", bbox_inches="tight")
