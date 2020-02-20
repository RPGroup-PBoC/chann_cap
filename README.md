<p align="center">
  <img src="logo.png">
</p>


# First-principles prediction of the information processing capacity of a simple genetic circuit 
Welcome to the GitHub repository for the channel capacity project! This
repository serves as a record for the experimental and theoretical work
described in the publication "*First-principles prediction of the information
processing capacity of a simple genetic circuit*" 

## Branches

This repository contains two main branches -- `master` and `gh-pages`. The
branch `master` which you are reading right now is the primary branch for the
project. In here you will find all of the polished and unpolished code used for
all the calculations and figure generation in the paper. The `gh-pages` branch
contains all of the [website files](https://www.rpgroup.caltech.edu/chann_cap/index.html).
What the branch `master` does not contain are the data files for the project.
But you can download such datasets from the links in the [website](https://www.rpgroup.caltech.edu/chann_cap/code).
Please see individual directories for more information.

## Installation
The intend of this repository is to make every step of the publication
completely transparent and reproducible. The project involved a significant
amount of home-grown Python code that we wrapped as a module `chann_cap`. To
install the package first you need to make sure that you have all of the
required dependencies. To check for this you can use 
[`pip`](pypi.org/project/pip) by executing the following command:

``` pip install -r requirements.txt ```

Once you have all of the packages installed locally, you can install our custom
module by running the following command:

``` pip install -e ./ ```

When installed, a new folder `chann_cap.egg-info` will be
installed. This folder is required for the executing of the code in this
repository.