# Logbook

This markdown file is a late addition to the project. I'll keep track of all
what I'm working on the project in order to have a record and a better track of
what I have already tried and what work or didn't work.

## 2019/12

**2019/12/19**
- Today I worked through the derivation of the one-state bursty gene expression
  following Charlotte's notes. After brainstorming with Muir about this idea,
  it became it's own little project. So look for a future paper on bursty
  transcription!

**2019/12/01**
- Today I cleaned the figures from the mRNA distribution MCMC fit notebook and
  generated independent scripts for each of them.
- I also finished the section that talks about the skewness and its systematic
  deviation with the data.
- I then cleaned the figures from the MaxEnt notebook and generated individual
  scripts for all these.

## 2019/11

**2019/11/27**
- Today I explored the idea that there could be a systematic deviation when
  cells binomially partition their content given that two cells in the
  population are perfectly correlated. But from numerical analysis I guess this
  hypothesis doesn't make sense anymore.
- I also worked on the Gillespie simulations that I wrote before. Cleaning the
  code and re-running the simulations with the 100% established parameters.
  This will definitely be added to the SI.

**2019/11/26**
- Today I hit a wall with respect to the Poisson Gaussian model. I tried a
  couple of things, such as simple relationships between moments of the
  distribution, but I didn't get a clean picture that explains the deviation.
  See `src/theory/sandbox/poisson_gaussian_noise.ipynb` for more details on
  what I tried.
- I also started working on analytical results related to the binomial
  partitioning of the proteins. I have an idea that all events after the
  binomial partitioning are not i.i.d. since in principle **two** daughter
  cells are perfectly correlated.

**2019/11/25**
- Today I worked on the Poisson-Gaussian noise model. The algebra gets messy,
  I'll need to use `sympy` to work through it.

