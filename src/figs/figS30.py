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

df_micro = pd.read_csv(f'{homedir}/data/csv_microscopy/' + 
                       '20181003_O2_RBS1027_IPTG_titration_microscopy.csv',
                       comment='#')

# Extract the data from the experimental strain
df_exp = df_micro[df_micro.rbs == "RBS1027"]
# run the bootstrap sample for a single fraction of the data
MI, samp_size = ccutils.channcap.channcap_bootstrap(df_exp, 200, 100, 0.5)
x, y = ccutils.stats.ecdf(MI)
plt.scatter(x, y, edgecolors="none")
plt.xlabel("channel capacity (bits)")
plt.ylabel("ECDF")
plt.margins(0.01)

plt.savefig(figdir + "figS30.pdf", bbox_inches="tight")

