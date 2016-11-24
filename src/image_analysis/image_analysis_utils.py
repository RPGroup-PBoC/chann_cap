import os
import glob
import pickle
import datetime
# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.special

# Useful plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Image analysis libraries
import skimage.io
import skimage.filters
import skimage.segmentation
import scipy.ndimage

#=============================================================================== 
# SEGMENTATION                    
#=============================================================================== 
def find_zero_crossings(im, selem, thresh):
    """
    This  function computes the gradients in pixel values of an image after
    applying a sobel filter to a given image. This  function is later used in
    the Laplacian of Gaussian cell segmenter (log_segmentation) function. The
    arguments are as follows.
    
    im = image to be filtered. 
    selem = structural element used to compute gradients. 
    thresh = threshold to define gradients. 
    Credit : Griffin Chure
    """

    #apply a maximum and minimum filter to the image. 
    im_max = scipy.ndimage.filters.maximum_filter(im, footprint=selem)
    im_min = scipy.ndimage.filters.minimum_filter(im, footprint=selem)

    #Compute the gradients using a sobel filter. 
    im_filt = skimage.filters.sobel(im)

    #Find the zero crossings. 
    zero_cross = (((im >=0) & (im_min < 0)) | ((im <= 0) & (im_max > 0)))\
            & (im_filt >= thresh)
    
    return zero_cross

#=============================================================================== 
def log_segmentation(im, selem, thresh=0.001, radius=2.0, clear_border=True):
    """
    This function computes the Laplacian of a gaussian filtered image and
    detects object edges as regions which cross zero in the derivative. The
    arguments are as follows:
    im = fluorescence image to be filtered and segmented.
    radius = radius for gaussian filter.
    selem = structural element to be applied for laplacian calculation.
    thresh = threshold to define gradients
    """
     
    #Ensure that the provided image is a float. 
    im_float = im_to_float(im)
    
    #Subtract background to fix illumination issues.
    im_gauss = skimage.filters.gaussian(im_float, 20.0)
    im_float = im_float - im_gauss

    #Compute the LoG filter of the image. 
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, radius)

    #Using find_zero_crossings, identify the edges of objects.
    edges = find_zero_crossings(im_LoG, selem, thresh)

    #Skeletonize the edges to a line with a single pixel width.
    skel_im = skimage.morphology.skeletonize(edges)

    #Fill the holes to generate binary image.
    im_fill = scipy.ndimage.morphology.binary_fill_holes(skel_im)

    #Remove small objects and objects touching border. 
    im_final = skimage.morphology.remove_small_objects(im_fill)
    if clear_border==True:
        im_final = skimage.segmentation.clear_border(im_final, buffer_size=5)

    #Return the labeled image. 
    return im_final

#============================================================================== 
# Plotting style
#============================================================================== 
def set_plotting_style():
    """
    Formats plotting enviroment to that used in Physical Biology of the Cell,
    2nd edition. To format all plots within a script, simply execute
    `mwc_induction_utils.set_plotting_style() in the preamble.
    """
    rc = {'lines.linewidth': 2,
          'axes.labelsize': 16,
          'axes.titlesize': 18,
          'axes.facecolor': '#E3DCD0',
          'xtick.labelsize': 13,
          'ytick.labelsize': 13,
          'font.family': 'Lucida Sans Unicode',
          'grid.linestyle': ':',
          'grid.linewidth': 1.5,
          'grid.color': '#ffffff',
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.fontsize': 11}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('darkgrid', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)

#============================================================================== 
# Define function to compute ECDF
#============================================================================== 
def ecdf(data):
    """
    Computes the empirical cumulative distribution function (ECDF)
    of a given set of 1D data.

    Parameters
    ----------
    data : 1d-array
        Data from which the ECDF will be computed.

    Returns
    -------
    x, y : 1d-arrays
        The sorted data (x) and the ECDF (y) of the data.
    """

    return np.sort(data), np.arange(len(data))/len(data)

#============================================================================== 
# Generic thermodynamic functions
#============================================================================== 

def p_act(C, ka, ki, epsilon=4.5, logC=False):
    '''
    Returns the probability of a lac repressor being in the active state, i.e.
    able to bind the promoter as a function of the ligand concentration.
    
    Parameters
    ----------
    C : array-like.
        concentration(s) of ligand at which evaluate the function.
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.
    epsilon : float.
        energetic barrier between the inactive and the active state.
    logC : Bool.
        boolean indicating if the concentration is given in log scale
    
    Returns
    -------
    p_act : float. 
        The probability of the repressor being in the active state.
    '''
    C = np.array(C)
    if logC:
        C = 10**C
        
    return (1 + C / ka)**2 / \
            ((1 + C / ka)**2 + np.exp(-epsilon) * (1 + C / ki)**2)

#============================================================================== 

def fold_change(iptg, ka, ki, epsilon, R, epsilon_r,
                    quaternary_state=2, nonspec_sites=4.6E6):
    '''
    Returns the gene expression fold change according to the
    thermodynamic model with the extension that takes into account the
    effect of the inducer.

    Parameter
    ---------
    iptg : array-like.
        Concentrations of inducer on which to evaluate the function
    ka, ki : float.
        dissociation constants for the active and inactive states respectively
        in the MWC model of the lac repressor.    
    epsilon : float.
        Energy difference between the active and the inactive state
    R : array-like.
        Repressor copy number for each of the strains. The length of
        this array should be equal to the iptg array. If only one value
        of the repressor is given it is asssume that all the data points
        should be evaluated with the same repressor copy number
    epsilon_r : array-like
        Repressor binding energy. The length of this array
        should be equal to the iptg array. If only one value of the
        binding energy is given it is asssume that all the data points
        should be evaluated with the same repressor copy number
    quaternary_state: int
        Prefactor in front of R in fold-change. Default is 2
        indicating that there are two functional heads per repressor molecule.
        This value must not be zero.
    nonspec_sites : int
        Number of nonspecific binding sites in the system.
        This value must be greater than 0.

    Returns
    -------
    fold_change : float.
        Gene expression fold change as dictated by the thermodynamic model.

    Raises
    ------
    ValueError
        Thrown if any entry of the IPTG vector, number of repressors,
        quaternary prefactor, or number of nonspecific binding sites is
        negative. This is also thrown if the quaternary
        state  or number of nonspecific binding sites is 0.


   '''
    return (1 + quaternary_state * R / nonspec_sites *
            p_act(iptg, ka, ki, epsilon) * (1 + np.exp(-epsilon)) *
            np.exp(-epsilon_r))**-1
