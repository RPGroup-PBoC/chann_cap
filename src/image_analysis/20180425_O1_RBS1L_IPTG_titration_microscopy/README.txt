# ---
# status: accepted
# ---
# 
# # Description
# IPTG titration of the O1 - RBS1L strain.
# 
# | | |
# |-|-|
# | __Date__ | 2018-04-25 |
# | __Equipment__ | Artemis Nikon Microscope |
# | __User__ | mrazomej |
# 
# ## Strain infromation
# | Genotype | plasmid | Host Strain | Shorthand |
# | :------- | :------ | :---------- | :-------- |
# | `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
# | `galK<>25-O1+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
# | `galK<>25-O1+11-YFP; ; ybcN::3*1-RBS1L-lacI` | `pZS3-mCherry` | HG105 | `RBS1L` |
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
# 2. mCherry : 10 ms
# 3. YFP : 17 ms
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
# 
# This is the first dataset taken with the new plate holder. Given the stability
# I was able to mark the positions and then take all images automatically. That
# means that the structure of this data set is completely different. There is
# a single folder per pad and a stacked image of each of the positions marked
# for this particular pad.
# I also happened to discover a little bit late the autofocus feature. So most
# of the pads were not taken with automatic focus, meaning that they could
# be slightly defocus. By eye it didn't seem to be a big issue, but let's keep
# this in mind for this particular dataset.
# Also given the new setup I wasn't able to take the YFP profile images, so I
# will use the images taken in 20180424.
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
# **ECDF (RBS1L)**
# 
# ![](outdir/exp_fluor_ecdf.png)
# 
# **fold-change**
# 
# ![](outdir/fold_change.png)
