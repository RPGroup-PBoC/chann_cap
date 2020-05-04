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

# Select RBS1027 day 1 to start the data exploration
df_group = df_micro[df_micro.rbs == 'RBS1027'].groupby('IPTG_uM')

# Extract concentrations
concentrations = df_micro.IPTG_uM.unique()

# Plot distributions coming from microscopy
# Decide color
colors = sns.color_palette("Blues_r", len(concentrations))

fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True)

# Set the nice scientific notation for the y axis of the histograms
ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(\
                             useMathText=True, 
                             useOffset=False))
ax[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(\
                             useMathText=True, 
                             useOffset=False))

# Set the number of bins for the histograms
nbins = 20 
# Initialize array to save the mean fluorescence
mean_fl = []

# Loop through each group
for i, (g, data) in enumerate(df_group):
    # Histogram plot
    # Add the filling to the histogram
    n, bins, patches = ax[0].hist(data.intensity, nbins,
                                  density=1, histtype='stepfilled', alpha=0.4,
                                  label=str(g)+ r' $\mu$M', facecolor=colors[i],
                                  linewidth=1)
    # Add a black outline for each histogram
    n, bins, patches = ax[0].hist(data.intensity, nbins,
                                density=1, histtype='stepfilled', 
                                label='', edgecolor='k',
                               linewidth=1.5, facecolor='none')
    # Save the mean fluorescence 
    mean_fl.append(data.intensity.mean())
    
    # ECDF Plot
    x, y = ccutils.stats.ecdf(data.intensity)
    ax[1].plot(x, y, '.', label=str(g)+ r' $\mu$M', color=colors[i])

# Declare color map for legend
cmap = plt.cm.get_cmap('Blues_r', len(concentrations))
bounds = np.linspace(0, len(concentrations), len(concentrations) + 1)

# Plot a little triangle indicating the mean of each distribution
mean_plot = ax[0].scatter(mean_fl, [5E-4] * len(mean_fl), marker='v', s=200,
            c=np.arange(len(mean_fl)), cmap=cmap,
            edgecolor='k',
            linewidth=1.5)
# Generate a colorbar with the concentrations
cbar_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
cbar = fig.colorbar(mean_plot, cax=cbar_ax)
cbar.ax.get_yaxis().set_ticks([])
for j, r in enumerate(concentrations):
    if r == 0.1:
        r = str(r)
    else:
        r = str(int(r))
    cbar.ax.text(1, j / len(concentrations) + 1 / (2 * len(concentrations)),
                 r, ha='left', va='center',
                 transform = cbar_ax.transAxes, fontsize=6)
cbar.ax.get_yaxis().labelpad = 35
cbar.set_label(r'IPTG ($\mu$M)')

    
ax[0].set_ylim([0, 1E-3])
ax[0].set_ylabel('probability')
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
 
ax[1].margins(0.02)
ax[1].set_xlabel('fluorescence (a.u.)')
ax[1].set_ylabel('ECDF')

plt.figtext(0.0, .9, '(A)', fontsize=8)
plt.figtext(0.0, .46, '(B)', fontsize=8)

plt.subplots_adjust(hspace=0.06)
plt.savefig(figdir + "figS24.pdf", bbox_inches="tight")

