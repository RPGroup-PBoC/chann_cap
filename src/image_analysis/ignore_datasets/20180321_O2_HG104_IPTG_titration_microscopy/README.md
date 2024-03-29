---
status: rejected
reason: All the high concentrations are systematically above the prediction by a significant amount.
---

# Description
IPTG titration of the O2 - HG104 strain.

| | |
|-|-|
| __Date__ | 2018-03-21 |
| __Equipment__ | Artemis Nikon Microscope |
| __User__ | mrazomej |

## Strain infromation
| Genotype | plasmid | Host Strain | Shorthand |
| :------- | :------ | :---------- | :-------- |
| `galK<>25` | `pZS3-mCherry` | HG105 | `auto` |
| `galK<>25-O2+11-YFP` | `pZS3-mCherry` | HG105 | `delta` |
| `galK<>25-O2+11-YFP | `pZS3-mCherry` | HG104 | `HG104` |

## Titration series
| Inducer | Concentration |
| :------ | ------------: |
| IPTG | 0, 0.1, 5, 10, 25, 50, 75, 100, 250, 500, 1000, 5000 [µM] |

## Microscope settings

* 100x Oil objective
* Exposure time:
1. Brightfield : 2 ms
2. mCherry : 18 ms
3. YFP : 13 ms

## Experimental protocol

The strains were grown overnight in tubes in 3 mL of LB + spec + kan.
Next morning they were diluted 1:1000 into 0.5 mL of M9 + 0.5% glucose
in deep 96-well plates.
The auto and delta strains were grown without IPTG.
After 8 hours the cells were diluted 1:10 into M9 + glucose and imaged
using 2% agar pads also of M9 media.

## Notes & Observations
The 75uM is labeled as 100uM because of my distraction, and the 100uM is
therefore called 100realuMIPTG.
It was hard to get 30 images for the last 2 concentrations so there are
less pictures for these concentrations.

## Analysis files

**Example segmentation**

![](outdir/example_segmentation.png)

**ECDF (auto)**

![](outdir/auto_fluor_ecdf.png)

**ECDF (∆lacI)**

![](outdir/delta_fluor_ecdf.png)

**ECDF (RBS1027)**

![](outdir/exp_fluor_ecdf.png)

**fold-change**

![](outdir/fold_change.png)
