# Source code

In this folder you can find all of the code used to generate every computation
and figures in the project. The folder is divided into sub-directories to
further classify the function of each piece of code.

## `image_analysis`

This folder contains the analysis pipelines to process all of the raw
microscopy data. From the segmentation of the cells, to the extraction of
single-cell quantitative fluorescence measurements. There is one folder per
experiment.

## `channcap_exp`

This directory takes the output from the `image_analysis` pipelines and
computes the channel experimental capacity using the Blahut-Arimoto algorithm
for each of the data sets.

## `theory`

This folder contains all of the code used to compute quantities related to the
theoretical model for the simple-repression motif. From analytical calculations
done using `sympy`, to numerical integration of differential equations.

## `figs`

This folder contains plain `Python` scripts to reproduce all figures in the
paper, including all of the SI material and further figures that didn't make it
to the final publication but are useful for talks and different version of
visualizations.