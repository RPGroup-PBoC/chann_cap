#%%
import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import re
import git

# Import libraries to parallelize processes
from joblib import Parallel, delayed

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

# REAL DATA
read_files = glob.glob(f"{homedir}/data/csv_channcap_bootstrap/*bootstrap.csv")
df_bs = pd.concat(pd.read_csv(f, comment="#") for f in read_files)

# Group by the number of bins
df_group = df_bs.groupby(["date", "operator", "rbs", "bins"])

# Initialize data frame to save the I_oo estimates
df_cc = pd.DataFrame(columns=["date", "operator", "rbs", "bins", "channcap"])
for group, data in df_group:
    x = 1 / data.samp_size
    y = data.channcap_bs
    # Perform linear regression
    lin_reg = np.polyfit(x, y, deg=1)
    df_tmp = pd.Series(
        list(group) + [lin_reg[1]],
        index=["date", "operator", "rbs", "bins", "channcap"],
    )
    df_cc = df_cc.append(df_tmp, ignore_index=True)

# Convert date and bins into integer
df_cc[["date", "bins"]] = df_cc[["date", "bins"]].astype(int)

# Group by date
df_O2_1027 = df_cc[df_cc["date"] == 20181003]

# SHUFFLED DATA
read_files = glob.glob(
    f"{homedir}/data/csv_channcap_bootstrap/*bootstrap_shuffled.csv"
)
df_bs_rnd = pd.concat(pd.read_csv(f, comment="#") for f in read_files)

# Group by the number of bins
df_group = df_bs_rnd.groupby(["date", "operator", "rbs", "bins"])

# Initialize data frame to save the I_oo estimates
df_cc_shuff = pd.DataFrame(
    columns=["date", "operator", "rbs", "bins", "channcap"]
)
for group, data in df_group:
    x = 1 / data.samp_size
    y = data.channcap_bs
    # Perform linear regression
    lin_reg = np.polyfit(x, y, deg=1)
    df_tmp = pd.Series(
        list(group) + [lin_reg[1]],
        index=["date", "operator", "rbs", "bins", "channcap"],
    )
    df_cc_shuff = df_cc_shuff.append(df_tmp, ignore_index=True)

# Convert date and bins into integer
df_cc_shuff[["date", "bins"]] = df_cc_shuff[["date", "bins"]].astype(int)

# Group by date
df_O2_1027_shuff = df_cc_shuff[df_cc_shuff["date"] == 20181003]

# Initialize figure
fig, ax = plt.subplots(1, 1)
# Plot real data
ax.plot(df_O2_1027.bins, df_O2_1027.channcap, label="experimental data")
# Plot shuffled data
ax.plot(
    df_O2_1027_shuff.bins, df_O2_1027_shuff.channcap, label="shuffled data"
)

# Label axis
ax.set_xlabel("# bins")
ax.set_ylabel(r"channel capacity $I_\infty$ (bits)")

# Set x scale to log
ax.set_xscale("log")

# Add legend
plt.legend()

plt.savefig(figdir + "figS27.pdf", bbox_inches="tight")

