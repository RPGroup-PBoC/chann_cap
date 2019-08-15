#---
#status: accepted
#---
#
## Description
#Testing the idea that cells could have memory of their time
#in LB. Therefore cells for this experiment were grown at all
#stages in M9 + 0.5% glucose. Overnights were done on this 
#media as well as the experiment obviously.
#
#| | |
#|-|-|
#| __Date__ | 2019-08-14 |
#| __Equipment__ | Artemis Nikon Microscope |
#| __User__ | mrazomej |
#
### Strain infromation
#| Genotype | plasmid | Host Strain | Shorthand |
#| :------- | :------ | :---------- | :-------- |
#| `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
#| `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
#| `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG104 | `HG104` |
#| `galK<>25-O2+11-YFP; ybcN<>3*1-RBS1027` | `pZS3-mCherry` | HG105 | `RBS1027` |
#| `galK<>25-O2+11-YFP; ybcN<>3*1-RBS1L` | `pZS3-mCherry` | HG105 | `RBS1L` |
#
### Titration series
#| Inducer | Concentration |
#| :------ | ------------: |
#| IPTG | 0 |
#
### Microscope settings
#
#* 100x Oil objective
#* Exposure time:
#1. Brightfield : 50 ms
#2. mCherry : 75 ms
#3. YFP : 70 ms (1 Volt power)
#
### Experimental protocol
#
#The strains were grown overnight in tubes in 2 mL of M9 glucose + spec + kan.
#These tubes grew for > 16 hours since they were inoculated the day before 
#early in the morning.
#Next morning I diluted the strains 1:1000 into 3 mL of 
#fresh media.
#I then prepared a 2% agar pad with PBS buffer and image them with the
#usual imaging settings
#
### Notes & Observations
#These cells grew extended time in M9 to guarantee that they reached the
#expected steady state.
#I forgot to take the camera noise so I copied the one from 20190626.
#
### Analysis files
#
#**Example segmentation**
#
#![](outdir/example_segmentation.png)
#
#**ECDF (auto)**
#
#![](outdir/auto_fluor_ecdf.png)
#
#**ECDF (∆lacI)**
#
#![](outdir/delta_fluor_ecdf.png)
#
#**ECDF (HG104)**
#
#![](outdir/exp_fluor_ecdf.png)
#
#**fold-change**
#
#![](outdir/fold_change.png)
