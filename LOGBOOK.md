# Logbook

This markdown file is a late addition to the project. I'll keep track of all
what I'm working on the project in order to have a record and a better track of
what I have already tried and what work or didn't work.

## 2019/11

**2019/11/25**
- Today I worked on the Poisson-Gaussian noise model. The algebra gets messy,
  I'll need to use `sympy` to work through it.

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