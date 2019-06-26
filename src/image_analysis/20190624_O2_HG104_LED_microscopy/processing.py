import os
import glob

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.special

# Import the project utils
import sys
sys.path.insert(0, '../../../')
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
        # List strain directory
        pos = glob.glob(data_dir + '*' + st + '*_' + name +
                           'uMIPTG*')

        if len(pos) is not 0:
            print(name)
            # List all images with 1) BF, 2) YFP, 3) mCherry
            c1images = np.sort(glob.glob(pos[0] + '/*c1.tif'))
            c2images = np.sort(glob.glob(pos[0] + '/*c2.tif'))
            c3images = np.sort(glob.glob(pos[0] + '/*c3.tif'))

            # Select random image to print example segmentation
            ex_no = np.random.choice(np.arange(0, len(c1images) - 1))
            # Loop through images
            for z, im in enumerate(c1images):
                _ = skimage.io.imread(im)
                y = skimage.io.imread(c2images[z])
                m = skimage.io.imread(c3images[z])

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
                                                 intensity_image=y)
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
