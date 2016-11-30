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
BINDING_ENERGY = -13.9
REPRESSORS = (0, 0, 130)
IPDIST = 0.160  # in units of µm per pixel
STRAINS = ['auto', 'delta', 'RBS1027']
IPTG_RANGE = (0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000)

#============================================================================== 

# Define the data directory.
data_dir = '../../../data/microscopy/' + str(DATE) + '/'

# Glob the profile and noise images.
yfp_glob = glob.glob(data_dir + '*yfp_profile*/*.tif')
rfp_glob = glob.glob(data_dir + '*mCherry_profile*/*.tif')
noise_glob = glob.glob(data_dir + '*noise*/*.tif')

# Load the images as collections
yfp_profile = skimage.io.ImageCollection(yfp_glob)
rfp_profile = skimage.io.ImageCollection(rfp_glob)
noise_profile = skimage.io.ImageCollection(noise_glob)

# Need to split the noise profile image into the two channels
noise_rfp = [noise_profile[i][0] for i, _ in enumerate(noise_profile)]
noise_yfp = [noise_profile[i][1] for i, _ in enumerate(noise_profile)]

# Generate averages and plot them. 
rfp_avg = im_utils.average_stack(rfp_profile)
yfp_avg = im_utils.average_stack(yfp_profile)

rfp_noise = im_utils.average_stack(noise_rfp)
yfp_noise = im_utils.average_stack(noise_yfp)

with sns.axes_style('white'):
    fig, ax =  plt.subplots(2, 2, figsize=(6,6))
    ax = ax.ravel()
    ax[0].imshow(yfp_avg, cmap=plt.cm.viridis)
    ax[0].set_title('yfp profile')
    ax[1].imshow(rfp_avg, cmap=plt.cm.plasma)
    ax[1].set_title('rfp profile')
    ax[2].imshow(yfp_noise, cmap=plt.cm.Greens_r)
    ax[2].set_title('yfp noise')
    ax[3].imshow(rfp_noise, cmap=plt.cm.Reds_r)
    ax[3].set_title('rfp noise')
plt.tight_layout()
plt.savefig('./outdir/background_correction.png')

#============================================================================== 

# Iterate through each strain and concentration to make the dataframes.
dfs = []
ex_iptg = np.random.choice(IPTG_RANGE)
for i, st in enumerate(STRAINS):
    print(st)
    for j, iptg in enumerate(IPTG_RANGE):
        # Load the images
        if (iptg==0) & (st != 'RBS1027'):
            images = glob.glob(data_dir + '*' + st + '_*/*.tif')
            
        else:
            images = glob.glob(data_dir + '*' + st + '*_' + str(iptg) +
                           'uMIPTG*/*.ome.tif')
            
        if len(images) is not 0:
            ims = skimage.io.ImageCollection(images)
    
            for _, x in enumerate(ims):
                _, m, y = im_utils.ome_split(x)
                y_flat = im_utils.generate_flatfield(y, yfp_noise, yfp_avg)
    
                # Segment the mCherry channel.
                m_seg = im_utils.log_segmentation(m, label=True)
                if iptg == ex_iptg:
                    ex_seg = m_seg
                    ex_phase = _
    
                # Extract the measurements.
                im_df = im_utils.props_to_df(m_seg, physical_distance=IPDIST,
                                        intensity_image=y_flat)
    
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
df_im.to_csv('./outdir/' + str(DATE) + '_' + OPERATOR + '_' +\
               STRAINS[-1] + '_raw_segmentation.csv', index=False)

