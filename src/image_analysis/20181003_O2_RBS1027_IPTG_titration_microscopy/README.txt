# ---
# status: accepted
# ---
# 
# # Description
# IPTG titration of the O2 - RBS1027 strain.
# 
# | | |
# |-|-|
# | __Date__ | 2018-10-03 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O2+11-YFP; ybcN<>4*RBS1027-lacI` | `pZS3-mCherry` | HG105 | `RBS1027` |
# 
# ## Titration series
# | Inducer | Concentration |
# | :------ | ------------: |
# | IPTG | 0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000 [µM] |
# 
# ## Microscope settings
# 
# * 100x Oil objective
# * Exposure time:
# 1. Brightfield : 10 ms
# 2.  mCherry : 50 ms (power = 1 mV)
# 3. YFP : 14 ms
# 
# ## Experimental protocol
# 
# The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
# Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
# in deep 96-well plates.
# The auto and delta strains were grown without IPTG.
# After 8 hours the cells were diluted 1:10 into M9 + glucose and imaged
# using 2% agar pads also of M9 media.
# 
# ## Notes & Observations
# First data set taken in several months. HJ checked the microscope
# and both channels (YFP and mCherry) seemed to be well aligned.
# Since I used the plate holder I was able to mark several positions
# at once. At the beginning I divided it into two set of images of
# ≈ 15 and 10 positions, but by the end the microscope was stable enough
# so that I could mark all 25 positions at once.
# This data set was not saved as an ome.tif image stack. Instead each
# individual channel was saved as an independent image inside the folder
# for their corresponding position. Therefore the processing code
# had to be adapted to read and process these images
# 
# ## Analysis files
# 
# **Example segmentation**
# 
# ![](outdir/example_segmentation.png)
# 
# **ECDF (auto)**
# 
# ![](outdir/auto_fluor_ecdf.png)
# 
# **ECDF (∆lacI)**
# 
# ![](outdir/delta_fluor_ecdf.png)
# 
# **ECDF (RBS1027)**
# 
# ![](outdir/exp_fluor_ecdf.png)
# 
# **fold-change**
# 
# ![](outdir/fold_change.png)
