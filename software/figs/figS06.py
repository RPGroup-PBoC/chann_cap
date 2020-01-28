#%%
import os
import pickle
import cloudpickle
import itertools
import glob
import numpy as np
import scipy as sp
import pandas as pd
import git
import corner

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
datadir = f'{homedir}/data/mcmc/'

# %%
with open(f"{datadir}lacUV5_constitutive_mRNA_double_expo.pkl", "rb") as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# Initialize subplot
fig, axes = plt.subplots(3, 3, figsize=(3.5, 3.5))

# Draw the corner plot
fig = corner.corner(
    gauss_flatchain,
    bins=50,
    plot_contours=True,
    labels=[r"$k^{(p)}_{on}$", r"$k^{(p)}_{off}$", r"$r_m$"],
    fig=fig,
)


plt.savefig(figdir + 'figS06.pdf', bbox_inches='tight')
