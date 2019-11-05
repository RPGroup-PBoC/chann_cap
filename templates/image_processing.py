import os
import glob

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.special

# Import the project utils
import ccutils.image as im_utils
import ccutils.viz as viz_utils

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
viz_utils.set_plotting_style()

# =============================================================================
# METADATA
# =============================================================================

from metadata import *

# =============================================================================

# Define the data directory.
data_dir = '../../../data/microscopy/' + str(DATE) + '/'

# Glob the profile and noise images.
yfp_glob = glob.glob(data_dir + '*YFP_profile*/*/*.tif')
noise_glob = glob.glob(data_dir + '*noise*/*/*.tif')

# Load the images as collections
yfp_profile = skimage.io.ImageCollection(yfp_glob)
noise_profile = skimage.io.ImageCollection(noise_glob)

# Need to split the noise profile image into the two channels
noise_yfp = [noise_profile[i] for i, _ in enumerate(noise_profile)]

# Generate averages and plot them.
yfp_avg = im_utils.average_stack(yfp_profile)

yfp_noise = im_utils.average_stack(noise_yfp)

with sns.axes_style('white'):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax = ax.ravel()
    ax[0].imshow(yfp_avg, cmap=plt.cm.viridis)
    ax[0].set_title('yfp profile')
    ax[1].imshow(yfp_noise, cmap=plt.cm.Greens_r)
    ax[1].set_title('yfp noise')
plt.tight_layout()
plt.savefig('./outdir/background_correction.png')

# =============================================================================

# Iterate through each strain and concentration to make the dataframes.
dfs = []

# Select random IPTG and random strain to print the example segmentation
ex_iptg = np.random.choice(IPTG_RANGE)
ex_strain = STRAINS[-1]
for i, st in enumerate(STRAINS):
    print(st)
    for j, name in enumerate(IPTG_NAMES):
        iptg = IPTG_DICT[name]
        # List position directories
        pos = glob.glob(data_dir + '*' + st + '*_' + name +
                           'uMIPTG*/Pos*')

        if len(pos) is not 0:
            print(name)
            # Select random image to print example segmentation
            ex_no = np.random.choice(np.arange(0, len(pos) - 1))
            for z, x in enumerate(pos):
                # List all images with 1) BF, 2) TRITC, 3) YFP
                images = np.sort(glob.glob(x + '/*tif'))
                _ = skimage.io.imread(images[0])
                m = skimage.io.imread(images[1])
                y = skimage.io.imread(images[2])
                y_flat = im_utils.generate_flatfield(y, yfp_noise, yfp_avg)

                # Segment the mCherry channel.
                m_seg = im_utils.log_segmentation(m, label=True)

                # Print example segmentation for the random image
                if (st == ex_strain) & (iptg == ex_iptg) & (z == ex_no):
                    merge = im_utils.example_segmentation(m_seg, _, 10/IPDIST)
                    skimage.io.imsave('./outdir/example_segmentation.png',
                                      merge)

                # Extract the measurements.
                try:
                    im_df = im_utils.props_to_df(m_seg,
                                                 physical_distance=IPDIST,
                                                 intensity_image=y_flat)
                except ValueError:
                    break

                # Add strain and  IPTG concentration information.
                im_df.insert(0, 'IPTG_uM', iptg)
                im_df.insert(0, 'repressors', REPRESSORS[i])
                im_df.insert(0, 'rbs', st)
                im_df.insert(0, 'binding_energy', BINDING_ENERGY)
                im_df.insert(0, 'operator', OPERATOR)
                im_df.insert(0, 'username', USERNAME)
                im_df.insert(0, 'date', DATE)

                # Append the dataframe to the global list.
                dfs.append(im_df)

# Concatenate the dataframe
df_im = pd.concat(dfs, axis=0)
df_im.to_csv('./outdir/' + str(DATE) + '_' + OPERATOR + '_' +
             STRAINS[-1] + '_raw_segmentation.csv', index=False)
