import os
import glob
import pandas as pd

# Read the tidy-data frame
files = glob.glob('../../../data/csv_microscopy/*IPTG*csv')# + mwc_files
df_micro = pd.concat([pd.read_csv(f, comment='#') for f in files], sort=True)

# Remove data sets that are ignored because of problems with 
# the data quality
# NOTE: These data sets are kept in the repository for transparency, but they
# failed at one of our quality criteria
# (see README.txt file in microscopy folder)
ignore_files = [x for x in os.listdir('../ignore_datasets/')
                if 'microscopy' in x]
# Extract data from these files
ignore_dates = [int(x.split('_')[0]) for x in ignore_files]

# Remove these dates
df_micro = df_micro[~df_micro['date'].isin(ignore_dates)]

# Rename repressors to repressor
df_micro = df_micro.rename(columns={'repressors': 'repressor'})

# Export file
df_micro.to_csv('../../../data/csv_microscopy/single_cell_microscopy_data.csv',
                index=False)
