#%%
import numpy as np
import pandas as pd
import ccutils

#%%

# Define number of boostrap estimates
n_estimates = 25000

# Define percentiles to save
percentiles = [.01, .05, .10, .25, .50, .75, .90, .95, .99]

# Read single cell data
df_micro = pd.read_csv('../../../data/csv_microscopy/' +
                       'single_cell_microscopy_data.csv')

#%%
# group by date and by IPTG concentration
df_group = df_micro.groupby(['date'])

# Define names for columns in data frame
names = ['date', 'IPTG_uM','operator', 'binding_energy',
         'repressor', 'percentile',
         'fold_change', 'fold_change_lower', 'fold_change_upper',
         'noise', 'noise_lower', 'noise_upper']

# Initialize data frame to save the noise
df_noise = pd.DataFrame(columns=names)

# Loop through groups
for date, data in df_group:
    print(f'date: {date}')
    # Extract the autofluorescence
    I_auto = data[data.rbs == 'auto'].mean_intensity.values
    print('bootstrapping autofluorescence')
    # Perform bootstrap estimate of mean autofluorescence
    boots_auto = ccutils.stats.bootstrap_estimate(I_auto, np.mean, n_estimates)
    
    # Extract ∆lacI data
    data_delta = data[data.rbs == 'delta']
    # Initialize array to save bootstrap estimates of the background corrected
    # ∆lacI intensity
    boots_mean_delta = np.zeros(n_estimates)
    boots_std_delta = np.zeros(n_estimates)
    print('bootstrapping ∆lacI')
    # Loop through estimates
    for i in range(n_estimates):
        # Sample data
        sample = data_delta.sample(n=len(data_delta), replace=True)
        # Compute bootstrap estimates
        boots_mean_delta[i] = np.mean(sample.intensity.values -
                                 boots_auto[i] * sample.area.values)
        boots_std_delta[i] = np.std(sample.intensity.values -
                             boots_auto[i] * sample.area.values, ddof=1)
    # Remove negative values
    boots_mean_delta = boots_mean_delta[boots_mean_delta >= 0]
    boots_std_delta = boots_std_delta[boots_mean_delta >= 0]
    # Compute ∆lacI noise
    boots_noise_delta = boots_std_delta / boots_mean_delta
    # Loop through percentiles and save information
    for per in percentiles:
        # Compute percentile noise
        per_noise = ccutils.stats.hpd(boots_noise_delta, per)
        strain_info = [
            date,
            None,
            data_delta.operator.unique()[0],
            data_delta.binding_energy.unique()[0],
            0,
            per,
            None,
            None,
            None,
            np.median(boots_noise_delta),
            per_noise[0],
            per_noise[1]
        ]
        # Append to dataframe
        df_noise = df_noise.append(pd.Series(strain_info, index=names),
                                   ignore_index=True)

    # Group data by IPTG concentration 
    data_group = data[(data.rbs != 'auto') &
                      (data.rbs != 'delta')].groupby('IPTG_uM')

    # Loop through inducer concentrations
    for inducer, data_inducer in data_group:
        print(f'bootstrapping {inducer} µM')
        # Initialize array to save bootstrap estimates of the background
        # corrected 
        boots_mean_inducer = np.zeros(n_estimates)
        boots_std_inducer = np.zeros(n_estimates)
        # Loop through estimates
        for i in range(n_estimates):
            # Sample data
            sample = data_inducer.sample(n=len(data_inducer), replace=True)
            # Compute bootstrap estimates
            boots_mean_inducer[i] = np.mean(sample.intensity.values -
                                    boots_auto[i] * sample.area.values)
            boots_std_inducer[i] = np.std(sample.intensity.values -
                                    boots_auto[i] * sample.area.values, ddof=1)
        # Remove netative reads
        boots_mean_inducer = boots_mean_inducer[boots_mean_inducer >= 0]
        boots_std_inducer = boots_std_inducer[boots_mean_inducer >= 0]
        # Compute fold-change and noise
        boots_fc_inducer = boots_mean_inducer / boots_mean_delta
        boots_noise_inducer = boots_std_inducer / boots_mean_inducer

        # Loop through percentiles and save information
        for per in percentiles:
            # Compute percentile noise
            per_fc = ccutils.stats.hpd(boots_fc_inducer, per)
            per_noise = ccutils.stats.hpd(boots_noise_inducer, per)
            strain_info = [
                date,
                inducer,
                data_inducer.operator.unique()[0],
                data_inducer.binding_energy.unique()[0],
                data_inducer.repressor.unique()[0],
                per,
                np.median(boots_fc_inducer),
                per_fc[0],
                per_fc[1],
                np.median(boots_noise_inducer),
                per_noise[0],
                per_noise[1]
            ]
            # Append to dataframe
            df_noise = df_noise.append(pd.Series(strain_info, index=names),
                                    ignore_index=True)

# %%

# Export dataframe
df_noise.to_csv('../../../data/csv_microscopy/microscopy_noise_bootstrap.csv')
