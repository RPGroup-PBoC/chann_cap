In order to keep transparency all of the data sets taken for this project
are kept in the repository. But only the ones that passed our quality
criteria were used for the rest of the work. 

Here we will list the reason why each of the following data sets were ignored:

- 20161118_O2_RBS1027_IPTG_titration_microscopy
The data for the fold-change is completely off from the theoretical expectation

- 20161129_O2_RBS1027_IPTG_titration_microscopy
The data for the fold-change is completely off from the theoretical expectation

- 20180320_O2_HG104_IPTG_titration_microscopy
The dataset was taken with the microscope in TIRF mode without me knowing. That
is why the exposure time was so much longer. Since there are a lot of weird things
that happen with this is better to discard the dataset.

- 20180328_O3_RBS1027_IPTG_titration_microscopy
The strain didn't respond to IPTG. All the distributions overlap with each other
so probably there was contamination with the Auto strain.

- 20180330_O3_RBS1L_channel_capacity_experiment
The Delta strain at 0 uM IPTG was contaminated with auto, and when setting the
exposure with the Delta strain at 5000uM most of the other strains were oversaturated
So all of the strains were not measured correctly with respect to Delta and
therefore it doesn't pass our quality criteria
