# Libraries to acces and write data
import glob
import os

# Numerical workhorses
import numpy as np
import pandas as pd

# List files from IPTG titration
files = glob.glob('../../data/csv_microscopy/*IPTG*csv')

# Read the tidy-data frame
df_micro = pd.concat(pd.read_csv(f, comment='#') for f in files if 'Oid' not in f)

##  Remove data sets that are ignored because of problems with the data quality
##  NOTE: These data sets are kept in the repository for transparency, but they
##  failed at one of our quality criteria
##  (see README.txt file in microscopy folder)
ignore_files = [x for x in os.listdir('./ignore_datasets/')
                if 'microscopy' in x]

# Extract data from these files
ignore_dates = [int(x.split('_')[0]) for x in ignore_files]

# Remove these dates
df_micro = df_micro[~df_micro['date'].isin(ignore_dates)].reset_index()

# Export single-cell intensities
df_micro.to_csv('../../data/single_cell_intensities.csv')
