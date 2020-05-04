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
df_sim_mRNA = pd.read_csv(f'{datadir}gillespie_mRNA.csv')
# Group data by simulation number
df_group = df_sim_mRNA.groupby("sim_num")

# Initialize plot
fig = plt.figure()

# Define colors
colors = sns.color_palette("Paired", n_colors=2)
# Loop through each simulation
for group, data in df_group:
    plt.plot(
        data.time / 60,
        data.mRNA,
        "-",
        lw=0.3,
        alpha=0.05,
        color=colors[0],
        label="",
    )

# Compute mean mRNA
mean_mRNA = [data.mRNA.mean() for group, data in df_sim_mRNA.groupby("time")]
time_points = np.sort(df_sim_mRNA.time.unique()) / 60

# # Plot mean mRNA
plt.plot(
    time_points,
    mean_mRNA,
    "-",
    lw=2,
    color=colors[1],
    label=r"$\left\langle m(t) \right\rangle$",
)

# Group data frame by cell cycle
df_group = df_sim_mRNA.groupby("cycle")
# Loop through cycles
for i, (group, data) in enumerate(df_group):
    # Define the label only for the last cell cycle not to repeat in legend
    if group == df_sim_mRNA["cycle"].max():
        label_s = "single promoter"
        label_d = "two promoters"
    else:
        label_s = ""
        label_d = ""
    # Find index for one-promoter state
    idx = np.where(data.state == "single")[0]
    # Indicate states with two promoters
    plt.axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#e3dcd1",
        label=label_s,
    )

    # Find index for two-promoter state
    idx = np.where(data.state == "double")[0]
    # Indicate states with two promoters
    plt.axvspan(
        data.iloc[idx.min()]["time"] / 60,
        data.iloc[idx.max()]["time"] / 60,
        facecolor="#ffedce",
        label=label_d,
    )

# Set limits
plt.xlim(df_sim_mRNA["time"].min() / 60, df_sim_mRNA["time"].max() / 60)
# Label plot
plt.xlabel("time (min)")
plt.ylabel("mRNA/cell")
plt.legend(loc="upper right")

# Save figure
plt.tight_layout()
plt.savefig(figdir + "figS20.pdf", bbox_inches="tight")
