import sys
import pickle
import os
import glob
import re
import datetime
import itertools

# Our numerical workhorses
import numpy as np
from sympy import mpmath
import scipy.optimize
import scipy.special
import scipy.integrate
import pandas as pd

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import AstroPy for bining histogram data
from astropy.stats import knuth_bin_width

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Import the utils for this project
sys.path.insert(0, '../../theory/')
import chann_cap_utils as chann_cap

chann_cap.set_plotting_style()

#==============================================================================
# METADATA
#==============================================================================

DATE = 20161129
USERNAME = 'mrazomej'
OPERATOR = 'O2'
STRAIN = 'RBS1027'
REPRESSOR = 130
BINDING_ENERGY = -13.9

#============================================================================== 


