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

#============================================================================== 
# METADATA
#============================================================================== 

DATE = 20161118
USERNAME = 'mrazomej'
OPERATOR = 'O2'
STRAIN = 'RBS1027'
REPRESSOR = 130
BINDING_ENERGY = -13.9

#============================================================================== 
# Read data
df_im = pd.read_csv('./outdir/' + str(DATE) + '_' + OPERATOR + '_' +\
               STRAIN + '_raw_segmentation.csv')

#============================================================================== 
# Group by strain
df_group = df_im.groupby('rbs')

# Plot area and eccentricity ECDF
fig, ax = plt.subplots(1, 2, figsize=(8,4))
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

#============================================================================== 

# Apply the area and eccentricity bounds.
df_filt = df_im[(df_im.area > 0.5) & (df_im.area < 6.0) &
                     (df_im.eccentricity > 0.8)]
# Save file in the same directory as the summary plots
df_filt.to_csv('./outdir/' +\
               str(DATE) + '_' + OPERATOR + '_' +\
               STRAIN + '_IPTG_titration_microscopy.csv', index=False)

# Export file to data directory including the comments
filenames = ['./README.txt', './outdir/' +
             str(DATE) + '_' + OPERATOR + '_' +\
             STRAIN + '_IPTG_titration_microscopy.csv']

with open('../../../data/csv_microscopy/' + str(DATE) + '_' + OPERATOR + '_' +\
               STRAIN + '_IPTG_titration_microscopy.csv', 'w') as output:
    for fname in filenames:
        with open(fname) as infile:
            output.write(infile.read())

#============================================================================== 

# Compute mean intensity for auto and delta strains
mean_auto = df_filt[df_filt.rbs == 'auto'].mean_intensity.mean()
mean_delta = df_filt[df_filt.rbs == 'delta'].mean_intensity.mean()

# Group analysis strain by RBS
df_group = df_filt[df_filt.rbs == STRAIN].groupby('IPTG_uM')

# Initialize dataframe to save fold change
df_fc = pd.DataFrame(columns=['IPTG', 'fold_change'])
for group, data in df_group:
    fold_change = (data.mean_intensity.mean() - mean_auto) /\
                  (mean_delta - mean_auto)
    df_tmp =  pd.DataFrame([group, fold_change], index=['IPTG', 'fold_change']).T
    df_fc = pd.concat([df_fc, df_tmp], axis=0)

#============================================================================== 

# Compute the theoretical fold change
iptg = np.logspace(-2, 4, 100)
fc = im_utils.fold_change(iptg=iptg, ka=141.52, ki=0.56061, epsilon=4.5, 
                          R=REPRESSOR,  epsilon_r=BINDING_ENERGY)

# Plot the fold-change
plt.figure()
plt.plot(iptg, fc, label='theoretical fold-change')
plt.plot(df_fc.IPTG, df_fc.fold_change, marker='o', linewidth=0,
         label='microscopy data')
plt.xscale('log')
plt.legend(loc=0)
plt.xlabel(r'IPTG ($\mu$M)')
plt.ylabel(r'fold-change')
plt.savefig('./outdir/fold_change.png', bbox_inches='tight')

#============================================================================== 

# Plot nice histogram and ECDF of filtered data
concentrations = df_filt.IPTG_uM.unique()
colors = sns.color_palette("Blues_r", len(concentrations))

fig, ax = plt.subplots(2, 1, figsize=(6,5), sharex=True)

# Set the nice scientific notation for the y axis of the histograms
ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(\
                             useMathText=True, 
                             useOffset=False))
ax[0].xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(\
                             useMathText=True, 
                             useOffset=False))

# Group data frame by concentration
df_group = df_filt[df_filt.rbs == 'RBS1027'].groupby('IPTG_uM')

# initialize counter for colors
i = 0
mean_fl = []
for c, data in df_group:
    mean_int = data.mean_intensity
    mean_fl.append(mean_int.mean())
    # Histogram plot
    n, bins, patches = ax[0].hist(mean_int, 30,
                                normed=1, histtype='stepfilled', alpha=0.4,
                                label=str(c)+ r' $\mu$M', facecolor=colors[i],
                               linewidth=1)
    n, bins, patches = ax[0].hist(mean_int, 30,
                                normed=1, histtype='stepfilled', 
                                label='', edgecolor='k',
                               linewidth=1.5, facecolor='none')
    # ECDF Plot
    x, y = im_utils.ecdf(mean_int)
    ax[1].plot(x, y, '.', label=str(c)+ r' $\mu$M', color=colors[i])
    
    # Increase counter
    i += 1

# Declare color map for legend
cmap = plt.cm.get_cmap('Blues_r', len(concentrations))
bounds = np.linspace(0, len(concentrations), len(concentrations) + 1)

# # Plot a little triangle indicating the mean of each distribution
mean_plot = ax[0].scatter(mean_fl, [0.018] * len(mean_fl), marker='v', s=200,
            c=np.arange(len(mean_fl)), cmap=cmap,
            edgecolor='k',
            linewidth=1.5)
# Generate a colorbar with the concentrations
cbar_ax = fig.add_axes([0.95, 0.25, 0.03, 0.5])
cbar = fig.colorbar(mean_plot, cax=cbar_ax)
cbar.ax.get_yaxis().set_ticks([])
for j, c in enumerate(concentrations):
    if c == 0.1:
        c = str(c)
    else:
        c = str(int(c))
    cbar.ax.text(1, j / len(concentrations) + 1 / (2 * len(concentrations)),
                 c, ha='left', va='center',
                 transform = cbar_ax.transAxes, fontsize=12)
cbar.ax.get_yaxis().labelpad = 35
cbar.set_label(r'[inducer] ($\mu$M)')

ax[0].set_ylim(bottom=0, top=0.02)
ax[0].set_ylabel('probability')
ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
 
ax[1].margins(0.02)
ax[1].set_xlabel('fluorescence (a.u.)')
ax[1].set_ylabel('ECDF')
ax[1].set_xlim(right=5000)

plt.figtext(0.0, .9, 'A', fontsize=20)
plt.figtext(0.0, .46, 'B', fontsize=20)

plt.subplots_adjust(hspace=0.06)
plt.savefig('./outdir/fluor_ecdf.png', bbox_inches='tight')
