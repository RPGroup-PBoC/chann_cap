import os
import glob

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.special

# Import the project utils
import sys
sys.path.insert(0, '../')
import image_analysis_utils as im_utils

# Useful plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns

# Image analysis libraries
import skimage.io
import skimage.filters
import skimage.segmentation
import scipy.ndimage

# Set plotting style
im_utils.set_plotting_style()

# =============================================================================
# METADATA
# =============================================================================

from metadata import *

# =============================================================================
# Read data
df_im = pd.read_csv('./outdir/' + str(DATE) + '_' + OPERATOR + '_' +
                    STRAIN + '_raw_segmentation.csv')

# =============================================================================
# Group by strain
df_group = df_im.groupby('rbs')

# Plot area and eccentricity ECDF
fig, ax = plt.subplots(1, 2, figsize=(5, 3))
for group, data in df_group:
    area_ecdf = im_utils.ecdf(df_im.area.sample(frac=0.3))
    ecc_ecdf = im_utils.ecdf(df_im.eccentricity.sample(frac=0.3))
    ax[0].plot(area_ecdf[0], area_ecdf[1], marker='.', linewidth=0,
               label=group, alpha=0.5)
    ax[1].plot(ecc_ecdf[0], ecc_ecdf[1], marker='.', linewidth=0,
               label=group, alpha=0.5)

# Format plots
ax[0].legend(loc='lower right', title='strain')
ax[0].set_xlabel(r'area ($\mu$m$^2$)')
ax[0].set_ylabel('ECDF')
ax[0].margins(0.02)

ax[1].set_xlabel(r'eccentricity')
ax[1].set_ylabel('ECDF')
ax[1].margins(0.02)

plt.tight_layout()
plt.savefig('./outdir/ecdf.png', bbox_inches='tight')

# =============================================================================

# Apply the area and eccentricity bounds.
df_filt = df_im[(df_im.area > 0.5) & (df_im.area < 6.0) &
                (df_im.eccentricity > 0.8)]
# Add column of absolute intensity
df_filt.loc[:, 'intensity'] = df_filt.area * df_filt.mean_intensity

# Save file in the same directory as the summary plots
df_filt.to_csv('./outdir/' +
               str(DATE) + '_' + OPERATOR + '_' +
               STRAIN + '_IPTG_titration_microscopy.csv', index=False)

# Export file to data directory including the comments
filenames = ['./README.txt', './outdir/' +
             str(DATE) + '_' + OPERATOR + '_' +
             STRAIN + '_IPTG_titration_microscopy.csv']

with open('../../../data/csv_microscopy/' + str(DATE) + '_' + OPERATOR + '_' +
          STRAIN + '_IPTG_titration_microscopy.csv', 'w') as output:
    for fname in filenames:
        with open(fname) as infile:
            output.write(infile.read())

# =============================================================================

# Initialize dataframe to save fold change
df_fc = pd.DataFrame(columns=['IPTG', 'fold_change', 'auto_IPTG'])

# List the concentrations at which the Auto and Delta strain were measured
auto_iptg = df_filt[(df_filt.rbs == 'auto')]['IPTG_uM'].unique()
delta_iptg = df_filt[(df_filt.rbs == 'delta')]['IPTG_uM'].unique()
fold_change_inducer = np.intersect1d(auto_iptg, delta_iptg)

# Loop through each concentration at whic auto and delta were measured
for c in fold_change_inducer:
    # Extract the mean auto and mean delta
    mean_auto = df_filt[(df_filt.rbs == 'auto') &
                        (df_filt.IPTG_uM == c)].intensity.mean()
    mean_delta = df_filt[(df_filt.rbs == 'delta') &
                         (df_filt.IPTG_uM == c)].intensity.mean()

    # Group analysis strain by RBS
    df_group = df_filt[df_filt.rbs == STRAIN].groupby('IPTG_uM')

    # Loop through each concentration in the experimental strain
    for group, data in df_group:
        # Compute the fold change
        fold_change = (data.intensity.mean() - mean_auto)\
                              / (mean_delta - mean_auto)

        # Append it to the data frame
        df_tmp = pd.DataFrame([group, fold_change, c],
                              index=['IPTG', 'fold_change', 'auto_IPTG']).T
        df_fc = pd.concat([df_fc, df_tmp], axis=0)

# =============================================================================

# Compute the theoretical fold change
# Log scale
iptg = np.logspace(-1, 4, 100)
fc = im_utils.fold_change(iptg=iptg, ka=141.52, ki=0.56061, epsilon=4.5,
                          R=REPRESSOR,  epsilon_r=BINDING_ENERGY)
# Linear scale
iptg_lin = [0, 1E-1]
fc_lin = im_utils.fold_change(iptg=iptg_lin, ka=141.52, ki=0.56061,
                              epsilon=4.5,
                              R=REPRESSOR,  epsilon_r=BINDING_ENERGY)

# Initialize figure
plt.figure(figsize=(4, 3))
# Plot theoretical fold-change
# Log scale
plt.plot(iptg, fc, label='theoretical fold-change', color='black')
plt.plot(iptg_lin, fc_lin, label='', linestyle='--', color='black')


# Group experimental data by concentration at which auto and delta were
# measured
df_group = df_fc.groupby('auto_IPTG')

# Loop through each concentration
for group, data in df_group:
    # Plot experimental fold-change
    plt.plot(data.IPTG, data.fold_change, marker='v', linewidth=0,
             label=r'$\Delta$ inducer {:.0f} $\mu$M'.format(group))

plt.xscale('symlog', linthreshx=1E-1, linscalex=0.5)
plt.legend(loc='upper left')
plt.ylim(bottom=0)
plt.xlabel(r'IPTG ($\mu$M)')
plt.ylabel(r'fold-change')
plt.savefig('./outdir/fold_change.png', bbox_inches='tight')

# =============================================================================

# Plot nice histogram for each strain
for strain in STRAINS:
    # Extract the particular data for the strain
    df_filt_strain = df_filt[df_filt['rbs'] == strain]

    # List the unique concentrations for this strain
    concentrations = df_filt_strain.IPTG_uM.unique()

    # Set a color pallete for each concentration
    colors = sns.color_palette("Blues_r", n_colors=len(concentrations))

    # Initialize figure
    fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

    # Set the nice scientific notation for the y axis of the histograms
    ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
                                 useMathText=True,
                                 useOffset=False))
    ax[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(
                                 useMathText=True,
                                 useOffset=False))

    # Group data frame by concentration
    df_group = df_filt_strain.groupby('IPTG_uM')

    # Initialize list to save mean fluorescence
    mean_fl = []

    # Initialize list to save max probability
    max_prob = []

    for i, (c, data) in enumerate(df_group):
        # Extract mean intensities
        mean_int = data.intensity
        # Save mean of mean intensities
        mean_fl.append(mean_int.mean())
        # Histogram plot
        n, bins, patches = ax[0].hist(mean_int, 30,
                                      density=1, histtype='stepfilled',
                                      alpha=0.4,
                                      label=str(c) + r' $\mu$M',
                                      facecolor=colors[i],
                                      linewidth=1)
        # Save max count
        max_prob.append(max(n))

        # add edges to the histograms
        n, bins, patches = ax[0].hist(mean_int, 30,
                                      density=1, histtype='stepfilled',
                                      label='', edgecolor='k',
                                      linewidth=1.5, facecolor='none')
        # ECDF Plot
        x, y = im_utils.ecdf(mean_int)
        ax[1].plot(x, y, '.', label=str(c) + r' $\mu$M', color=colors[i])

    # Declare color map for legend
    cmap = mpl.colors.ListedColormap(colors)
    bounds = np.linspace(0, len(concentrations), len(concentrations) + 1)

    # Plot a little triangle indicating the mean of each distribution
    mean_plot = ax[0].scatter(mean_fl,
                              [max(max_prob) * 1.1] * len(mean_fl),
                              marker='v', s=200,
                              c=np.arange(len(mean_fl)), cmap=cmap,
                              edgecolor='k', linewidth=1.5)

    # Generate a colorbar with the concentrations
    cbar_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(mean_plot, cax=cbar_ax)
    # Remove axis labels
    cbar.ax.get_yaxis().set_ticks([])

    # Loop through concentrations and add my own labels
    for j, c in enumerate(concentrations):
        if c == 0.1:
            c = str(c)
        else:
            c = str(int(c))
            cbar.ax.text(1, j / len(concentrations) +
                         1 / (2 * len(concentrations)),
                         c, ha='left', va='center',
                         transform=cbar_ax.transAxes, fontsize=12)
    cbar.ax.get_yaxis().labelpad = 35
    cbar.set_label(r'[inducer] ($\mu$M)')

    ax[0].set_ylim(bottom=0, top=max(max_prob) * 1.2)
    ax[0].set_ylabel('probability')
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax[1].margins(0.01)
    ax[1].set_xlabel('fluorescence (a.u.)')
    ax[1].set_ylabel('ECDF')

    plt.figtext(0.0, .9, 'A', fontsize=20)
    plt.figtext(0.0, .46, 'B', fontsize=20)

    # Change strain name to have same name for all strains
    if strain == STRAIN:
        strain = 'exp'
    plt.subplots_adjust(hspace=0.06)
    plt.savefig('./outdir/' + strain + '_fluor_ecdf.png', bbox_inches='tight')
