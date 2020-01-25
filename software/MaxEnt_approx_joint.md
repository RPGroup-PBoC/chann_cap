# Maximum Entropy Approximation - Joint distribution $P(m, p)$

(c) 2020 Manuel Razo. This work is licensed under a [Creative Commons Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). All code contained herein is licensed under an [MIT license](https://opensource.org/licenses/MIT). 

---


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
import os
import itertools
import cloudpickle
import re
import glob
import git

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy as sp

# Import library to perform maximum entropy fits
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns
# Increase DPI of displayed figures
%config InlineBackend.figure_format = 'retina'

# Import the project utils
import ccutils

# Find home directory for repo
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir

tmpdir = f'{homedir}/tmp/'
figdir = f'{homedir}/fig/MaxEnt_approx_joint/'
datadir = f'{homedir}/data/csv_maxEnt_dist/'
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Set PBoC plotting format
ccutils.viz.set_plotting_style()
# Increase dpi
mpl.rcParams['figure.dpi'] = 110
```

### $\LaTeX$ macros

$\newcommand{kpon}{k^p_{\text{on}}}$
$\newcommand{kpoff}{k^p_{\text{off}}}$
$\newcommand{kron}{k^r_{\text{on}}}$
$\newcommand{kroff}{k^r_{\text{off}}}$
$\newcommand{rm}{r _m}$
$\newcommand{rp}{r _p}$
$\newcommand{gm}{\gamma _m}$
$\newcommand{gp}{\gamma _p}$
$\newcommand{mm}{\left\langle m \right\rangle}$
$\newcommand{foldchange}{\text{fold-change}}$
$\newcommand{ee}[1]{\left\langle #1 \right\rangle}$
$\newcommand{bb}[1]{\mathbf{#1}}$
$\newcommand{th}[1]{\text{th}}$

## The MaxEnt approximation.

Given the difficulty at solving chemical master equations (CME) there is an extensive repertoire of approximate methods to tackle the problem of solving these equations. A particularly interesting method uses the so-called moment-expansion and maximum entropy approach to approximate distributions given knowledge of some of the moments of the distribution.

To illustrate the principle let us focus on a univariate distribution $P_X(x)$.
The $n^{\text{th}}$ moment of the distribution for a discrete set of possible
values of $x$ is given by

\begin{equation}
  \ee{x^n} \equiv \sum_x x^n P_X(x).
  \label{eq_mom_ref}
  \tag{1}
\end{equation}

Now assume that we have knowledge of the first $m$ moments $\bb{\ee{x}}_m = (
\ee{x}, \ee{x^2}, \ldots, \ee{x^m} )$. The question is then how can we use this
information to build an estimator $P_H(x \mid \bb{\ee{x}}_m)$ of the
distribution
such that

\begin{equation}
  \lim_{m \rightarrow \infty} P_H(x \mid \bb{\ee{x}}_m) \rightarrow P_X(x),
  \tag{2}
\end{equation}
i.e. that the more moments we add to our approximation, the more the estimator
distribution converges to the real distribution.

The MaxEnt principle tells us that our best guess for this estimator is to build
it on the base of maximizing the Shannon entropy, constrained by the information
we have about these $m$ moments. The maximization of Shannon's entropy
guarantees that we are the least committed possible to information that we do
not posses. The Shannon entropy for an univariate discrete distribution is
given by

\begin{equation}
  H(x) \equiv - \sum_x P_X(x) \log P_X(x).
  \tag{3}
\end{equation}

For an optimization problem subject to constraints we make use of the method of
the Lagrange multipliers. For this we define the Lagrangian $\mathcal{L}(x)$ as
\begin{equation}
  \mathcal{L}(x) \equiv H(x) - \sum_{i=0}^m
  \left[ \lambda_i \left( \ee{x^i} - \sum_x x^i P_X(x) \right) \right],
  \tag{4}
\end{equation}
where $\lambda_i$ is the Lagrange multiplier associated with the $i^{\text{th}}$
moment. The inclusion of the zeroth moment is an additional constraint to
guarantee the normalization of the resulting distribution.

Since $P_X(x)$ has a finite set of discrete values if we take the derivative of
the Lagrangian with respect to $P_X(x)$ what this implies is that we chose a
particular value of $X = x$. Therefore from the sum over all possible $x$ values
only a single term survives. With this in mind we take the derivative of the
Lagrangian obtaining

\begin{equation}
  {d\mathcal{L} \over d P_X(x)} = -\log P_X(x) - 1 -
  \sum_{i=0}^m \lambda_i x^i.
  \tag{5}
\end{equation}

Equating this derivative to zero and solving for the distribution (that we now
start calling $P_H(x)$, our MaxEnt estimator) gives

\begin{equation}
  P_H(x) = \exp \left(- 1 - \sum_{i=0}^m \lambda_i x^i \right)
         ={1 \over \mathcal{Z}}
         \exp \left( - \sum_{i=1}^m \lambda_i x^i \right),
  \tag{6}
\end{equation}
where $\mathcal{Z}$ is the normalization constant that can be obtained by
substituting this solution into the normalization constraint. This results in

\begin{equation}
  \mathcal{Z} \equiv \exp\left( 1 + \lambda_0 \right) =
  \sum_x \exp \left( - \sum_{i=1}^m \lambda_i x^i \right).
  \tag{7}
\end{equation}

Eq. (6) is the general form of the MaxEnt distribution for a univariate
distribution. The computational challenge then consists in finding numerical
values for the Lagrange multipliers $\{ \lambda_i \}$ such that $P_H(x)$
satisfies our constraints. In other words, the Lagrange multipliers weight the
contribution of each term in the exponent such that when computing any of the
moments we recover the value of our constraint. Mathematically what this means
is that $P_H(x)$ must satisfy

\begin{equation}
  \sum_x x^n P_H(x) =
  \sum_x {x^n \over \mathcal{Z}}
  \exp \left( - \sum_{i=1}^m \lambda_i x^i \right) = \ee{x^n}.
  \tag{8}
\end{equation}

### Numerically estimating the probability distribution.

Given that we have a theoretical expectation of what the moments of the distribution are, we can use this approximation to generate an estimate for the entire distribution. The only limiting step is to numerically determine the value of the Lagrange multipliers $\lambda_i$.

Instead of directly estimating the distribution we will use the `maxentropy` package in Python to perform the fit.

In the following example we will fit the classic example of an unfair die with a mean value of $\ee{x} = 4.5$ as done in [this notebook](https://github.com/PythonCharmers/maxentropy/blob/master/notebooks/Loaded%20die%20example%20-%20skmaxent.ipynb).


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Fit a model p(x) for dice probabilities (x=1,...,6) with the
# single constraint E(X) = 4.5
def first_moment_die(x):
    return np.array(x)


# Put the constraint functions into an array
features = [first_moment_die]
# Write down the constraints (in this case mean of 4.5)
k = np.array([4.5])

# Define the sample space of the die (from 1 to 6)
samplespace = list(range(1, 7))

# Define the minimum entropy
model = MinDivergenceModel(features, samplespace)

# Change the dimensionality of the array
X = np.atleast_2d(k)
# Fit the model
model.fit(X)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



    MinDivergenceModel(algorithm='CG',
                       features=[<function first_moment_die at 0x1c1e003680>],
                       matrix_format='csr_matrix', prior_log_pdf=None,
                       samplespace=[1, 2, 3, 4, 5, 6], vectorized=False, verbose=0)



Let's look at the resulting distribution. Here we will plot two possible solutions consistent with the limited information. On the left we will plot a biased distribution with average $\ee{x} = 4.5$, and on the right we will show the resulting MaxEnt distribution.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# initialize figure
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True, sharey=True)

# Define probability distribution of the "wrong inference"
prob = [0, 0, 0, 0.5, 0.5, 0]
# Plot the "wrong" distribution
ax[0].bar(samplespace, prob)

# Plot the max ent distribution
ax[1].bar(samplespace, model.probdist())

# Label axis
ax[0].set_xlabel("die face")
ax[1].set_xlabel("die face")

ax[0].set_ylabel("probability")

# Set title for plots
ax[0].set_title(r"$\left\langle x \right\rangle = 4.5$")
ax[1].set_title(r"MaxEnt $\left\langle x \right\rangle = 4.5$")

# Add letter label to subplots
plt.figtext(0.1, 0.93, "(A)", fontsize=8)
plt.figtext(0.50, 0.93, "(B)", fontsize=8)

plt.subplots_adjust(wspace=0.05)

plt.savefig(figdir + "biased_die_dist.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_11_0.png)


Here we can see that given that we are only told that the average was 4.5 there is no explicit information about certain die faces values not being used whatsoever. So the distribution in (A) is commited to information that we do not posses about the die, making a biased inference. On the other hand since the MaxEnt distribution in (B) maximizes the Shannon entropy subject to our limited information, it is guaranteed to be the least biased with respect to information that we do not posses about the process.

## The mRNA and protein joint distribution $P(m, p)$

The MaxEnt principle can easily be extended to multivariate distributions. For
our particular case we are interested in the mRNA and protein joint distribution
$P(m, p)$. The definition of a moment $\ee{m^x p^y}$ is a natural extension of
\eref{eq_mom_ref} of the form

\begin{equation}
  \ee{m^x p^y} = \sum_m \sum_p m^x p^y P(m, p).
  \tag{9}
\end{equation}

As a consequence the MaxEnt joint distribution $P_H(m, p)$ is of the form

\begin{equation}
  P_H(m, p) = {1 \over \mathcal{Z}}
              \exp \left( - \sum_{(x,y)} \lambda_{(x,y)} m^x p^y \right),
  \tag{10}
\end{equation}
where $\lambda_{x,y}$ is the Lagrange multiplier associated with the moment
$\ee{m^x p^y}$, and again $\mathcal{Z}$ is the normalization constant given by

\begin{equation}
  \mathcal{Z} = \sum_m \sum_p
              \exp \left( - \sum_{(x,y)} \lambda_{(x,y)} m^x p^y \right).
  \tag{11}
\end{equation}
Note that the sum in the exponent is taken over all available $(x, y)$ pairs
that define the moment constraints for the distribution.

## The Bretthorst rescaling algorithm

The determination of the Lagrange multipliers suffer from a numerical under and
overflow problem due to the difference in magnitude between the constraints.
This becomes a problem when higher moments are taken into account. The resulting
numerical values for the Lagrange multipliers end up being separated by several
orders of magnitude. For routines such as Newton-Raphson or other minimization
algorithms that can be used to find these Lagrange multipliers these different
scales become problematic.

To get around this problem we implemented a variation to the algorithm due to G.
Larry Bretthorst, E.T. Jaynes' last student. With a very simple argument we can
show that linearly rescaling the constraints, the Lagrange multipliers and the
"rules" for how to compute each of the moments, i.e. each of the individual
products that go into the moment calculation, should converge to the same MaxEnt
distribution. In order to see this let's consider again an univariate
distribution $P_X(x)$ that we are trying to reconstruct given the first two
moments. The MaxEnt distribution can be written as

\begin{equation}
  P_H(x) = {1 \over \mathcal{Z}}
  \exp \left(- \lambda_1 x - \lambda_2 x^2 \right) =
  {1 \over \mathcal{Z}}
  \exp \left(- \lambda_1 x \right) \exp \left( - \lambda_2 x^2 \right).
  \tag{12}
\end{equation}
We can always rescale the terms in any way and obtain the same result. Let's say
that for some reason we want to rescale the quadratic terms by a factor $a$. We
can define a new Lagrange multiplier $\lambda_2' \equiv {\lambda_2 \over a}$
that compensates for the rescaling of the terms, obtaining

\begin{equation}
  P_H(x) = {1 \over \mathcal{Z}}
  \exp \left(- \lambda_1 x \right) \exp \left( - \lambda_2' ax^2 \right).
  \tag{13}
\end{equation}
Computationally it might be more efficient to find the numerical value of
$\lambda_2'$ rather than $\lambda_2$ maybe because it is of the same order of
magnitude as $\lambda_1$. Then we can always multiply $\lambda_2'$ by $a$ to
obtain back the constraint for our quadratic term. What this means is that that
we can always rescale the MaxEnt problem to make it numerically more stable,
then we can rescale back to obtain the value of the Lagrange multipliers.

Bretthorst algorithm goes even further by further transforming the constraints
and the variables to make the constraints orthogonal, making the computation
much more effective. We now explain the implementation of the algorithm to our
joint distribution of interest $P(m, p)$.

### Algorithm implementation

Let the matrix $\bb{A}$ contain all the rules used to compute the moments that
serve as constraints, where each entry is of the form
\begin{equation}
  A_{ij} = m_i^{x_j} \cdot p_i^{y_j},
  \tag{14}
\end{equation}
i.e. the $i^{th}$ entry of our sample space consisting of of the product of all
possible pairs ($m, p$) elevated to the appropriate powers $x$ and $y$
associated with the $j^{th}$ constraint. Let also $\bb{v}$ be a vector
containing all the constraints with each entry of the form

\begin{equation}
  v_j = \ee{m^{x_j} p^{y_j}}.
  \tag{15}
\end{equation}

That means that the Lagrangian $\mathcal{L}$ to be used for this constrained
maximization problem takes the form

\begin{equation}
  \mathcal{L} = -\sum_i P_i \ln P_i + \lambda_0 \left( 1 - \sum_i P_i \right)
  + \sum_{j>0} \lambda_j \left( v_j - \sum_i A_{ij} P_i \right),
  \tag{16}
\end{equation}
where $\lambda_0$ is the Lagrange multiplier associated with the normalization
constraint, and $\lambda_j$ is the Lagrange multiplier associated with the
$j^{th}$ constraint.

With this notation in hand we now proceed to rescale the problem. The first
step consists of rescaling the rules to compute the entries of matrix $\bb{A}$
as

\begin{equation}
  A_{ij}' = {A_{ij} \over G_j},
  \tag{17}
\end{equation}
where $G_j$ serves to normalize the moments such that all the Lagrange
multipliers are of the same order of magnitude. This normalization satisfies

\begin{equation}
G_j^2 = \sum_i A_{ij}^2,
\tag{18}
\end{equation}
or in terms of our particular problem
\begin{equation}
G_j^2 = \sum_m \sum_p \left( m^{x_j} p^{y_j} \right)^2.
\tag{19}
\end{equation}

Since we rescale the rules to compute the constraints, the constraints must
also be rescaled simply as
\begin{equation}
v_j' = \ee{m^{x_j} p^{y_j}}' = {\ee{m^{x_j} p^{y_j}} \over G_j}.
\tag{20}
\end{equation}

The Lagrange multipliers must compensate this rescaling since at the end of the
day the probability must add up to the same value. Therefore we rescale the
$\lambda_j$ terms as as

\begin{equation}
\lambda_j' = \lambda_j G_j.
\tag{21}
\end{equation}

This rescaling by itself would already improve the algorithm convergence since
now all the Lagrange multipliers would not have drastically different values.
Bretthorst proposes another linear transformation to make the optimization
routine even more efficient. For this we generate orthogonal constraints that
make Newton-Raphson and similar routines converge faster. The transformation is
as follows

\begin{equation}
  A_{ik}'' = \sum_j {e}_{jk} A_{ij}',
  \tag{22}
\end{equation}
for the entires of matrix $\bb{A}$, and

\begin{equation}
  v_k'' = \sum_j {e}_{jk} u_j',
  \tag{23}
\end{equation}
for entires of the constraint vector $\bb{v}$, finally

\begin{equation}
  \lambda_k'' = \sum_j {e}_{jk} \beta_j,
  \tag{24}
\end{equation}
for the Lagrange multipliers. Here ${e}_{jk}$ is the $j^{th}$ component
of the $k^{th}$ eigenvector of the matrix $\bb{E}$ with entries

\begin{equation}
  {E}_{kj} = \sum_i {A}_{ik}' {A}_{ij}'.
  \tag{25}
\end{equation}

This transformation guarantees that the matrix $\bb{A}''$ has the property

\begin{equation}
  \sum_i A_{ij}'' A_{jk}'' = \beta_j \delta_{jk},
  \tag{26}
\end{equation}
where $\beta_j$ is the $j^{th}$ eigenvalue of the matrix $\bb{E}$ and
$\delta_{jk}$ is the delta function. What this means is that, as desired, the
constraints are orthogonal to each other, improving the algorithm convergence
speed.

There is an extra step that we will add to the algorithm to facilitate convergence. For the `maxentropy` package is better to have constraints that are all close to 1. Since the previous orthogonalization was generating very small constraints we will divide by the smallest of these transformed constraints. Again since it is a linear transformation this should not affect the final result.

#### Defining a function to compute the MaxEnt distribution

Now that we got the algorithm let's define a simple function that takes as inputs the `features` matrix, the list of constraints and the sample space, and it computes the MaxEnt distribution using the Bretthorst algorithm to then return the list of Lagrange Multipliers.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
def MaxEnt_bretthorst(
    constraints,
    features,
    algorithm="BFGS",
    tol=1e-4,
    paramtol=5e-5,
    maxiter=1000,
):
    """
    Computes the maximum entropy distribution given a list of constraints and a
    matrix with the features associated with each of the constraints using
    the maxentropy package. In particular this function rescales the problem
    according to the Bretthorst algorithm to fascilitate the gradient-based
    convergence to the value of the Lagrange multipliers.

    Parameters
    ----------
    constraints : array-like.
        List of constraints (moments of the distribution).
    features : 2D-array. shape = len(samplespace) x len(constraints)
        List of "rules" used to compute the constraints from the sample space.
        Each column has a rule associated and each row is the computation of
        such rule over the sample space.
        Example:
            If the ith rule is of the form m**x * p**y, then the ith column
            of features takes every possible pair (m, p) and computes such
            sample space.
    algorithm : string. Default = 'BFGS'
        Algorithm to be used by the maxentropy package.
        See maxentropy.BaseModel for more information.
    tol : float.
        Tolerance criteria for the convergence of the algorithm.
        See maxentropy.BaseModel for more information.
    paramtol : float.
        Tolerance criteria for the convergence of the parameters.
        See maxentropy.BaseModel for more information.
    maxiter : float.
        Maximum number of iterations on the optimization procedure.
        See maxentropy.BaseModel for more information.

    Returns
    -------
    Lagrange : array-like. lenght = len(constraints)
        List of Lagrange multipliers associated with each of the constraints.
    """
    # Define a dummy samplespace that we don't need since we are giving the
    # matrix of pre-computed features, but the maxentropy package still
    # requires it.
    samplespace = np.zeros(np.max(features.shape))

    # # First rescaling # #

    # Compute the factor to be used to re-scale the problem
    rescale_factor = np.sqrt(np.sum(features ** 2, axis=1))

    # Re-scale the features
    features_rescale = np.divide(features.T, rescale_factor).T

    # Re-scale constraints
    constraints_rescale = constraints / rescale_factor

    # # Orthogonalization # #

    # Compute the matrix from which the eigenvectors must be extracted
    features_mat = np.dot(features_rescale, features_rescale.T)

    # Compute the eigenvectors of the matrix
    trans_eigvals, trans_eigvects = np.linalg.eig(features_mat)

    # Transform the features with the matrix of eigenvectors
    features_trans = np.dot(trans_eigvects, features_rescale)

    # Transform the features with the constraints of eigenvectors
    constraints_trans = np.dot(trans_eigvects, constraints_rescale)

    # # Second rescaling # #

    # Find the absolute value of the smallest constraint that will be used
    # to rescale again the problem
    scale_min = np.min(np.abs(constraints_trans))

    # Scale by dividing by this minimum value to have features and
    # constraints close to 1
    features_trans_scale = features_trans / scale_min
    constraints_trans_scale = constraints_trans / scale_min

    # # Computing the MaxEnt distribution # #

    # Define the minimum entropy
    model = MinDivergenceModel(features_trans_scale, samplespace)

    # Set model features
    model.algorithm = algorithm
    model.tol = tol
    model.paramstol = paramtol
    model.maxiter = maxiter
    model.callingback = (
        True  # TBH I don't know what this does but it is needed
    )
    # for the damn thing to work

    # Change the dimensionality of the array
    # step required by the maxentropy package.
    X = np.reshape(constraints_trans_scale, (1, -1))

    # Fit the model
    model.fit(X)

    # # Transform back the Lagrange multipliers # #

    # Extract params
    params = model.params

    # peroform first rescaling
    params = params / scale_min

    # Transform back from the orthogonalization
    params = np.dot(np.linalg.inv(trans_eigvects), params)

    # Perform second rescaling
    params = params / rescale_factor

    return params
```

Once we have the Lagrange multipliers we can compute the probability mass function for whichever entry. Let's define a vectorized function that returns a 2D matrix probability distribution given an mRNA and protein sample space along with al ist of Lagrange multipliers.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
def maxEnt_from_lagrange(
    mRNA,
    protein,
    lagrange,
    exponents=[(1, 0), (2, 0), (3, 0), (0, 1), (0, 2), (1, 1)],
    log=False,
):
    """
    Computes the mRNA and protein joint distribution P(m, p) as approximated
    by the MaxEnt methodology given a set of Lagrange multipliers.
    Parameters
    ----------
    mRNA, protein : array-like.
        Sample space for both the mRNA and the protein.
    lagrange : array-like.
        Array containing the value of the Lagrange multipliers associated
        with each of the constraints.
    exponents : list. leng(exponents) == len(lagrange)
        List containing the exponents associated with each constraint.
        For example a constraint of the form <m**3> has an entry (3, 0)
        while a constraint of the form <m * p> has an entry (1, 1).
    log : bool. Default = False
        Boolean indicating if the log probability should be returned.
    Returns
    -------
    Pmp : 2D-array. len(mRNA) x len(protein)
        2D MaxEnt distribution.
    """
    # Generate grid of points
    mm, pp = np.meshgrid(mRNA, protein)

    # Initialize 3D array to save operations associated with each lagrange
    # multiplier
    operations = np.zeros([len(lagrange), len(protein), len(mRNA)])

    # Compute operations associated with each Lagrange Multiplier
    for i, expo in enumerate(exponents):
        operations[i, :, :] = lagrange[i] * mm ** expo[0] * pp ** expo[1]

    # check if the log probability should be returned
    if log:
        return np.sum(operations, axis=0) - sp.special.logsumexp(
            np.sum(operations, axis=0)
        )
    else:
        return np.exp(
            np.sum(operations, axis=0)
            - sp.special.logsumexp(np.sum(operations, axis=0))
        )
```

## Computing the Maximum entropy distribution

In order to test these functions we will use the moment inferences computed from integrating the moment equations and averaging over the cell cycle.

Let's import the data frame containing these moments.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Load moments for multi-promoter level
df_constraints = pd.read_csv(
    f'{homedir}/data/csv_maxEnt_dist/MaxEnt_multi_prom_constraints.csv'
)

# Remove the zeroth moment column
df_constraints = df_constraints.drop(labels="m0p0", axis=1)
```

Let's test the inference for a single set of parameters. We'll use only the protein moments of a single strain. So we need to extract the moments from the dataframe, define the range of protein values that we will use.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract protein moments in constraints
prot_mom = [x for x in df_constraints.columns if "m0" in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r"\d+", s))) for s in prot_mom]

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 10e4)

# Generate sample space as a list of pairs using itertools.
samplespace = list(itertools.product(mRNA_space, protein_space))

# Initialize matrix to save all the features that are fed to the
# maxentropy function
features = np.zeros([len(moments), len(samplespace)])

# Loop through constraints and compute features
for i, mom in enumerate(moments):
    features[i, :] = [ccutils.maxent.feature_fn(x, mom) for x in samplespace]
```

We will now run the algorithm to infer the Lagrange multipliers for a single distribution.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
df = df_constraints[
    (df_constraints.operator == "O2")
    & (df_constraints.repressor == 0)
    & (df_constraints.inducer_uM == 0)
]

# Define column names containing the constraints used to fit the distribution
constraints_names = ["m" + str(m[0]) + "p" + str(m[1]) for m in moments]

# Extract constraints (and convert to series)
constraints = df.T.squeeze().loc[constraints_names]

# Perform MaxEnt computation
# We use the Powell method because despite being slower it is more
# robust than the other implementations.
Lagrange = MaxEnt_bretthorst(
    constraints,
    features,
    algorithm="Powell",
    tol=1e-5,
    paramtol=1e-5,
    maxiter=10000,
)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    /Users/razo/anaconda3/lib/python3.7/site-packages/scipy/optimize/_minimize.py:500: RuntimeWarning: Method Powell does not use gradient information (jac).
      RuntimeWarning)


To make sure that the inference makes sense let's construct the distribution from the moments and make sure that it is normalized and it has the expected mean. This is not a super robust check, but we'll do further analysis later in this notebook.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Reconstruct distribution from Lagrange multipliers
pdist = maxEnt_from_lagrange(
    mRNA_space,
    protein_space,
    Lagrange,
    exponents=moments
)

# Compute normalization
pnorm = pdist.flatten().sum()
print(f'Normalization = {pnorm}')

# Compute the mean
pmean  = (pdist.flatten() * protein_space).sum()
print(f'Original mean = {float(df.m0p1)}')
print(f'MaxEnt mean = {pmean}')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    Normalization = 0.999999999999998
    Original mean = 7732.5821922890145
    MaxEnt mean = 7730.022444837335


This looks accurate enough. So let's now systematically compute the Lagrange multipliers for different parameter sets.

## Working at the protein level

We begin our inferences at the protein level. All these inferences, computationally expensive are done in `./scripts/maxent_protein_dist.py` and save as a tidy dataframe that we will load here. These computations were done using the `joblib` library to parallelize our efforts.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(
    f"{datadir}MaxEnt_Lagrange_mult_protein.csv"
)
df_maxEnt.head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>inducer_uM</th>
      <th>lambda_m0p1</th>
      <th>lambda_m0p2</th>
      <th>lambda_m0p3</th>
      <th>lambda_m0p4</th>
      <th>lambda_m0p5</th>
      <th>lambda_m0p6</th>
      <th>...</th>
      <th>m3p0</th>
      <th>m3p1</th>
      <th>m3p2</th>
      <th>m3p3</th>
      <th>m4p0</th>
      <th>m4p1</th>
      <th>m4p2</th>
      <th>m5p0</th>
      <th>m5p1</th>
      <th>m6p0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.002918</td>
      <td>-1.853543e-07</td>
      <td>-3.375754e-13</td>
      <td>6.208877e-18</td>
      <td>-8.239236e-23</td>
      <td>-3.409273e-28</td>
      <td>...</td>
      <td>8592.936386</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.002918</td>
      <td>-1.853543e-07</td>
      <td>-3.375754e-13</td>
      <td>6.208877e-18</td>
      <td>-8.239236e-23</td>
      <td>-3.409273e-28</td>
      <td>...</td>
      <td>8592.936386</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.002918</td>
      <td>-1.853543e-07</td>
      <td>-3.375754e-13</td>
      <td>6.208877e-18</td>
      <td>-8.239236e-23</td>
      <td>-3.409273e-28</td>
      <td>...</td>
      <td>8592.936386</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.002918</td>
      <td>-1.853543e-07</td>
      <td>-3.375754e-13</td>
      <td>6.208877e-18</td>
      <td>-8.239236e-23</td>
      <td>-3.409273e-28</td>
      <td>...</td>
      <td>8592.936386</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.002918</td>
      <td>-1.853543e-07</td>
      <td>-3.375754e-13</td>
      <td>6.208877e-18</td>
      <td>-8.239236e-23</td>
      <td>-3.409273e-28</td>
      <td>...</td>
      <td>8592.936386</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



Let's look at some of these distributions. First we will take a look to some PMFs


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define operators to be included
operators = ["O1", "O2", "O3"]

# Define repressors to be included
repressors = [22, 260, 1740]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define color for operators
# Generate list of colors
col_list = ["Blues", "Oranges", "Greens"]
col_dict = dict(zip(operators, col_list))

# Define binstep for plot, meaning how often to plot
# an entry
binstep = 100

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 1.3e4)

# Initialize plot
fig, ax = plt.subplots(
    len(repressors), len(operators), figsize=(5, 5), sharex=True, sharey=True
)

# Define displacement
displacement = 5e-5

# Loop through operators
for j, op in enumerate(operators):
    # Loop through repressors
    for i, rep in enumerate(repressors):

        # Extract the multipliers for a specific strain
        df_sample = df_maxEnt[
            (df_maxEnt.operator == op)
            & (df_maxEnt.repressor == rep)
            & (df_maxEnt.inducer_uM.isin(inducer))
        ]

        # Group multipliers by inducer concentration
        df_group = df_sample.groupby("inducer_uM", sort=True)

        # Extract and invert groups to start from higher to lower
        groups = np.flip([group for group, data in df_group])

        # Define colors for plot
        colors = sns.color_palette(col_dict[op], n_colors=len(df_group) + 1)

        # Initialize matrix to save probability distributions
        Pp = np.zeros([len(df_group), len(protein_space)])

        # Loop through each of the entries
        for k, group in enumerate(groups):
            data = df_group.get_group(group)

            # Select the Lagrange multipliers
            lagrange_sample = data.loc[
                :, [col for col in data.columns if "lambda" in col]
            ].values[0]

            # Compute distribution from Lagrange multipliers values
            Pp[k, :] = ccutils.maxent.maxEnt_from_lagrange(
                mRNA_space, protein_space, lagrange_sample, exponents=moments
            ).T

            # Generate PMF plot
            ax[i, j].plot(
                protein_space[0::binstep],
                Pp[k, 0::binstep] + k * displacement,
                drawstyle="steps",
                lw=1,
                color="k",
                zorder=len(df_group) * 2 - (2 * k),
            )
            # Fill between each histogram
            ax[i, j].fill_between(
                protein_space[0::binstep],
                Pp[k, 0::binstep] + k * displacement,
                [displacement * k] * len(protein_space[0::binstep]),
                color=colors[k],
                alpha=1,
                step="pre",
                zorder=len(df_group) * 2 - (2 * k + 1),
            )

        # Add x label to lower plots
        if i == 2:
            ax[i, j].set_xlabel("protein / cell")

        # Add y label to left plots
        if j == 0:
            ax[i, j].set_ylabel("[IPTG] ($\mu$M)")

        # Add operator top of colums
        if i == 0:
            label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(
                df_sample.binding_energy.unique()[0]
            )
            ax[i, j].set_title(label, bbox=dict(facecolor="#ffedce"))

        # Add repressor copy number to right plots
        if j == 2:
            # Generate twin axis
            axtwin = ax[i, j].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(rep),
                bbox=dict(facecolor="#ffedce"),
            )
            # Remove residual ticks from the original left axis
            ax[i, j].tick_params(color="w", width=0)

# Change lim
ax[0, 0].set_ylim([-3e-5, 7.5e-4 + len(df_group) * displacement])
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Set y axis ticks
yticks = np.arange(len(df_group)) * displacement
yticklabels = [int(x) for x in groups]

ax[0, 0].yaxis.set_ticks(yticks)
ax[0, 0].yaxis.set_ticklabels(yticklabels)

# Set x axis ticks
xticks = [0, 5e3, 1e4, 1.5e4]
ax[0, 0].xaxis.set_ticks(xticks)

# Save figure
plt.savefig(figdir + "PMF_grid_joyplot_protein.pdf", bbox_inches="tight")
plt.savefig(figdir + "PMF_grid_joyplot_protein.svg", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_38_0.png)


## Comparison with experimental data

Now that we have predictions of the protein distributions let's compare those with the predictions from the experimental single-cell fluorescence distributions.

First we need to read the data into memory.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Remove these dates
df_micro = pd.read_csv(
    f"{homedir}/data/csv_microscopy/single_cell_microscopy_data.csv"
)

df_micro[["date", "operator", "rbs", "mean_intensity", "intensity"]].head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>operator</th>
      <th>rbs</th>
      <th>mean_intensity</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>88.876915</td>
      <td>502.830035</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>99.759342</td>
      <td>393.291230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>94.213193</td>
      <td>552.315421</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>92.993102</td>
      <td>426.131591</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>94.591855</td>
      <td>455.251678</td>
    </tr>
  </tbody>
</table>
</div>



Since these values are in arbitrary units of fluorescence we need to find a way to relate them to the theoretical predictions. An optimal way to normalize these measurements is by computing the fold-change in gene expression. This is just the ratio of the intensity to the mean intensity of the $\Delta lacI$ strain, and is defined to be between 0 and 1.

Each experiment, i.e. each date in the data set, was taken in principle with different conditions of exposure time and laser intensity. Nevertheless each day an autofluorescence control and a $\Delta lacI$ strain were image. So we will group by date, and compute the fold-change for each of the measurements.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# group df by date
df_group = df_micro.groupby("date")

# loop through dates
for group, data in df_group:
    # Extract mean autofluorescence
    mean_auto = data[data.rbs == "auto"].mean_intensity.mean()
    # Extract ∆lacI data
    delta = data[data.rbs == "delta"]
    mean_delta = (delta.intensity - delta.area * mean_auto).mean()
    # Compute fold-change
    fc = (data.intensity - data.area * mean_auto) / (mean_delta - mean_auto)
    # Add result to original dataframe
    df_micro.loc[fc.index, "fold_change"] = fc


df_micro[
    ["date", "operator", "rbs", "mean_intensity", "intensity", "fold_change"]
].head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>operator</th>
      <th>rbs</th>
      <th>mean_intensity</th>
      <th>intensity</th>
      <th>fold_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>88.876915</td>
      <td>502.830035</td>
      <td>-0.009750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>99.759342</td>
      <td>393.291230</td>
      <td>-0.000335</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>94.213193</td>
      <td>552.315421</td>
      <td>-0.005393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>92.993102</td>
      <td>426.131591</td>
      <td>-0.005057</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20181018</td>
      <td>O2</td>
      <td>auto</td>
      <td>94.591855</td>
      <td>455.251678</td>
      <td>-0.004153</td>
    </tr>
  </tbody>
</table>
</div>



Let's now take a look at the distributions. We will start with the $\Delta lacI$ strains. Since we still would have issues with the bin size of a histogram, we will compare distirbutions using an ECDF that doesn't present such problems. We will split the operators because there could be systematic changes in the unregulated promoters as we have seen with the sort-seq data.

These experimental measurements will be compared with theoretical predictions to see how well our input-output function built from the kinetic model is able to reproduce the experimental data. Since we normalized the x-axis to be fold-change, we will compute the same quantity for our input-output function by just dividing by the mean protein number of the unregulated case.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 2.2e4)

# Extract the multipliers for a specific strain
df_maxEnt_delta = df_maxEnt[
    (df_maxEnt.operator == "O1")
    & (df_maxEnt.repressor == 0)
    & (df_maxEnt.inducer_uM == 0)
]

# Select the Lagrange multipliers
lagrange_sample = df_maxEnt_delta.loc[
    :, [col for col in df_maxEnt_delta.columns if "lambda" in col]
].values[0]

# Compute distribution from Lagrange multipliers values
Pp = ccutils.maxent.maxEnt_from_lagrange(
    mRNA_space,
    protein_space,
    lagrange_sample,
    exponents=moments
).T

# Compute mean protein copy number
mean_delta_p = np.sum(protein_space * Pp)

# Transform protein_space into fold-change
fc_space = protein_space / mean_delta_p

##  Plot ECDF for experimental data
# Keep only data for ∆lacI
df_delta = df_micro[df_micro.rbs == "delta"]

# Group data by operator
df_group = df_delta.groupby("operator")

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)

# Define colors for operators
col_list = ["Blues_r", "Reds_r", "Greens_r"]
col_dict = dict(zip(("O1", "O2", "O3"), col_list))

# Loop through operators
for i, (group, data) in enumerate(df_group):
    # Group data by date
    data_group = data.groupby("date")
    # Generate list of colors
    colors = sns.color_palette(col_dict[group], n_colors=len(data_group))

    # Loop through dates
    for j, (g, d) in enumerate(data_group):
        # Generate ECDF
        x, y = ccutils.stats.ecdf(d.fold_change)
        # Plot ECDF
        ax[i].plot(
            x[::10],
            y[::10],
            lw=0,
            marker=".",
            color=colors[j],
            alpha=0.3,
            label="",
        )

    # Label x axis
    ax[i].set_xlabel("fold-change")
    # Set title
    label = r"operator {:s}".format(group)
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))

    # Plot theoretical prediction
    ax[i].plot(
        fc_space[0::100],
        np.cumsum(Pp)[0::100],
        linestyle="--",
        color="k",
        linewidth=1.5,
        label="theory",
    )

    # Add fake data point for legend
    ax[i].plot([], [], lw=0, marker=".", color=colors[0], label="microscopy")
    # Add legend
    ax[i].legend()

# Label y axis of left plot
ax[0].set_ylabel("ECDF")

# Change limit
ax[0].set_xlim(right=3)

# Change spacing between plots
plt.subplots_adjust(wspace=0.05)
plt.savefig(figdir + "ECDF_unreg_theory_experiment.pdf")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_45_0.png)


There is a systematic deviation between the theoretical predictions and the compilation of experimental data. In a sense as we saw before when computing the noise, there is a larger variability in the experimental data compared with the model.

Let's take a look at the regulated cases.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define repressor copy number and operator
rep = [22, 260, 1740]
op = "O2"

# Define binstep for plot
binstep = 10
binstep_theory = 100

# Define colors
colors = sns.color_palette("Oranges_r", n_colors=len(inducer) + 2)

# Initialize plot
fig, ax = plt.subplots(
    len(rep), len(inducer), figsize=(7, 4.5), sharex=True, sharey=True
)

# Loop through repressor copy numbers
for j, r in enumerate(rep):
    # Loop through concentrations
    for i, c in enumerate(inducer):
        # Extract data
        data = df_micro[
            (df_micro.repressor == r)
            & (df_micro.operator == op)
            & (df_micro.IPTG_uM == c)
        ]

        # generate experimental ECDF
        x, y = ccutils.stats.ecdf(data.fold_change)

        # Plot ECDF
        ax[j, i].plot(
            x[::binstep],
            y[::binstep],
            color=colors[i],
            alpha=1,
            lw=3,
            label="{:.0f}".format(c),
        )

        # Extract lagrange multiplieres
        df_me = df_maxEnt[
            (df_maxEnt.operator == op)
            & (df_maxEnt.repressor == r)
            & (df_maxEnt.inducer_uM == c)
        ]

        lagrange_sample = df_me.loc[
            :, [col for col in df_me.columns if "lambda" in col]
        ].values[0]

        # Compute distribution from Lagrange multipliers values
        Pp = ccutils.maxent.maxEnt_from_lagrange(
            mRNA_space, protein_space, lagrange_sample, exponents=moments
        ).T

        # Plot theoretical prediction
        ax[j, i].plot(
            fc_space[0::binstep_theory],
            np.cumsum(Pp)[0::binstep_theory],
            linestyle="--",
            color="k",
            alpha=0.75,
            linewidth=1.5,
            label="",
        )

        # Label x axis
        if j == len(rep) - 1:
            ax[j, i].set_xlabel("fold-change")

        # Label y axis
        if i == 0:
            ax[j, i].set_ylabel("ECDF")

        # Add title to plot
        if j == 0:
            ax[j, i].set_title(
                r"{:.0f} ($\mu M$)".format(c),
                color="white",
                bbox=dict(facecolor=colors[i]),
            )

        # Add repressor copy number to right plots
        if i == len(inducer) - 1:
            # Generate twin axis
            axtwin = ax[j, i].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(r), bbox=dict(facecolor="#ffedce")
            )
            # Remove residual ticks from the original left axis
            ax[j, i].tick_params(color="w", width=0)

fig.suptitle(
    r"$\Delta\epsilon_r = {:.1f}\; k_BT$".format(-13.9),
    bbox=dict(facecolor="#ffedce"),
    size=10,
)
plt.subplots_adjust(hspace=0.05, wspace=0.02)
plt.savefig(figdir + "ECDF_O2_theory_experiment.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_47_0.png)


Again, there is that systematic deviation where the experimental data has a larger spread compared to the theoretical values.

Let's look at the other two operators


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define repressor copy number and operator
rep = [22, 260, 1740]
op = "O1"

# Define binstep for plot
binstep = 10
binstep_theory = 100

# Define colors
colors = sns.color_palette("Blues_r", n_colors=len(inducer) + 2)

# Initialize plot
fig, ax = plt.subplots(
    len(rep), len(inducer), figsize=(7, 4.5), sharex=True, sharey=True
)

# Loop through repressor copy numbers
for j, r in enumerate(rep):
    # Loop through concentrations
    for i, c in enumerate(inducer):
        # Extract data
        data = df_micro[
            (df_micro.repressor == r)
            & (df_micro.operator == op)
            & (df_micro.IPTG_uM == c)
        ]

        # generate experimental ECDF
        x, y = ccutils.stats.ecdf(data.fold_change)

        # Plot ECDF
        ax[j, i].plot(
            x[::binstep],
            y[::binstep],
            color=colors[i],
            alpha=1,
            lw=3,
            label="{:.0f}".format(c),
        )

        # Extract lagrange multiplieres
        df_me = df_maxEnt[
            (df_maxEnt.operator == op)
            & (df_maxEnt.repressor == r)
            & (df_maxEnt.inducer_uM == c)
        ]

        lagrange_sample = df_me.loc[
            :, [col for col in df_me.columns if "lambda" in col]
        ].values[0]

        # Compute distribution from Lagrange multipliers values
        Pp = ccutils.maxent.maxEnt_from_lagrange(
            mRNA_space, protein_space, lagrange_sample, exponents=moments
        ).T

        # Plot theoretical prediction
        ax[j, i].plot(
            fc_space[0::binstep_theory],
            np.cumsum(Pp)[0::binstep_theory],
            linestyle="--",
            color="k",
            alpha=0.75,
            linewidth=1.5,
            label="",
        )

        # Label x axis
        if j == len(rep) - 1:
            ax[j, i].set_xlabel("fold-change")

        # Label y axis
        if i == 0:
            ax[j, i].set_ylabel("ECDF")

        # Add title to plot
        if j == 0:
            ax[j, i].set_title(
                r"{:.0f} ($\mu M$)".format(c),
                color="white",
                bbox=dict(facecolor=colors[i]),
            )

        # Add repressor copy number to right plots
        if i == len(inducer) - 1:
            # Generate twin axis
            axtwin = ax[j, i].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(r), bbox=dict(facecolor="#ffedce")
            )
            # Remove residual ticks from the original left axis
            ax[j, i].tick_params(color="w", width=0)

fig.suptitle(
    r"$\Delta\epsilon_r = {:.1f}\; k_BT$".format(-15.3),
    bbox=dict(facecolor="#ffedce"),
    size=10,
)
plt.subplots_adjust(hspace=0.05, wspace=0.02)
plt.savefig(figdir + "ECDF_O1_theory_experiment.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_49_0.png)


Finally O3


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define repressor copy number and operator
rep = [22, 260, 1740]
op = "O3"

# Define binstep for plot
binstep = 10
binstep_theory = 100

# Define colors
colors = sns.color_palette("Greens_r", n_colors=len(inducer) + 2)

# Initialize plot
fig, ax = plt.subplots(
    len(rep), len(inducer), figsize=(7, 4.5), sharex=True, sharey=True
)

# Loop through repressor copy numbers
for j, r in enumerate(rep):
    # Loop through concentrations
    for i, c in enumerate(inducer):
        # Extract data
        data = df_micro[
            (df_micro.repressor == r)
            & (df_micro.operator == op)
            & (df_micro.IPTG_uM == c)
        ]

        # generate experimental ECDF
        x, y = ccutils.stats.ecdf(data.fold_change)

        # Plot ECDF
        ax[j, i].plot(
            x[::binstep],
            y[::binstep],
            color=colors[i],
            alpha=1,
            lw=3,
            label="{:.0f}".format(c),
        )

        # Extract lagrange multiplieres
        df_me = df_maxEnt[
            (df_maxEnt.operator == op)
            & (df_maxEnt.repressor == r)
            & (df_maxEnt.inducer_uM == c)
        ]

        lagrange_sample = df_me.loc[
            :, [col for col in df_me.columns if "lambda" in col]
        ].values[0]

        # Compute distribution from Lagrange multipliers values
        Pp = ccutils.maxent.maxEnt_from_lagrange(
            mRNA_space, protein_space, lagrange_sample, exponents=moments
        ).T

        # Plot theoretical prediction
        ax[j, i].plot(
            fc_space[0::binstep_theory],
            np.cumsum(Pp)[0::binstep_theory],
            linestyle="--",
            color="k",
            alpha=0.75,
            linewidth=1.5,
            label="",
        )

        # Label x axis
        if j == len(rep) - 1:
            ax[j, i].set_xlabel("fold-change")

        # Label y axis
        if i == 0:
            ax[j, i].set_ylabel("ECDF")

        # Add title to plot
        if j == 0:
            ax[j, i].set_title(
                r"{:.0f} ($\mu M$)".format(c),
                color="white",
                bbox=dict(facecolor=colors[i]),
            )

        # Add repressor copy number to right plots
        if i == len(inducer) - 1:
            # Generate twin axis
            axtwin = ax[j, i].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(r), bbox=dict(facecolor="#ffedce")
            )
            # Remove residual ticks from the original left axis
            ax[j, i].tick_params(color="w", width=0)

fig.suptitle(
    r"$\Delta\epsilon_r = {:.1f}\; k_BT$".format(-9.7),
    bbox=dict(facecolor="#ffedce"),
    size=10,
)
plt.subplots_adjust(hspace=0.05, wspace=0.02)
plt.savefig(figdir + "ECDF_O3_theory_experiment.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_51_0.png)


Now let's try to plot all of these comparisons for all operators on a single large plot.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define operators to be included
operators = ["O1", "O2", "O3"]
energies = [-15.3, -13.9, -9.7]

# Define repressor to be included
repressor = [22, 260, 1740]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define color for operators
# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
col_dict = dict(zip(operators, col_list))

# Initialize figure
fig = plt.figure(figsize=(7, 15))
# Define outer grid
outer = mpl.gridspec.GridSpec(len(operators), 1, hspace=0.3)

# Loop through operators
for k, op in enumerate(operators):
    # Initialize inner grid
    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        len(rep), len(inducer), subplot_spec=outer[k], wspace=0.02, hspace=0.05
    )

    # Define colors
    colors = sns.color_palette(col_dict[op], n_colors=len(inducer) + 2)

    # Loop through repressor copy numbers
    for j, r in enumerate(rep):
        # Loop through concentrations
        for i, c in enumerate(inducer):
            # Initialize subplots
            ax = plt.Subplot(fig, inner[j, i])

            # Add subplot to figure
            fig.add_subplot(ax)

            # Extract data
            data = df_micro[
                (df_micro.repressor == r)
                & (df_micro.operator == op)
                & (df_micro.IPTG_uM == c)
            ]

            # generate experimental ECDF
            x, y = ccutils.stats.ecdf(data.fold_change)

            # Plot ECDF
            ax.plot(
                x[::binstep],
                y[::binstep],
                color=colors[i],
                alpha=1,
                lw=3,
                label="{:.0f}".format(c),
            )

            # Extract lagrange multiplieres
            df_me = df_maxEnt[
                (df_maxEnt.operator == op)
                & (df_maxEnt.repressor == r)
                & (df_maxEnt.inducer_uM == c)
            ]

            lagrange_sample = df_me.loc[
                :, [col for col in df_me.columns if "lambda" in col]
            ].values[0]

            # Compute distribution from Lagrange multipliers values
            Pp = ccutils.maxent.maxEnt_from_lagrange(
                mRNA_space, protein_space, lagrange_sample, exponents=moments
            ).T

            # Plot theoretical prediction
            ax.plot(
                fc_space[0::binstep_theory],
                np.cumsum(Pp)[0::binstep_theory],
                linestyle="--",
                color="k",
                alpha=0.75,
                linewidth=1.5,
                label="",
            )

            # Label x axis
            if j == len(rep) - 1:
                ax.set_xlabel("fold-change")

            # Label y axis
            if i == 0:
                ax.set_ylabel("ECDF")

            # Add title to plot
            if j == 0:
                ax.set_title(
                    r"{:.0f} ($\mu M$)".format(c),
                    color="white",
                    bbox=dict(facecolor=colors[i]),
                )

            # Remove x ticks and y ticks from middle plots
            if i != 0:
                ax.set_yticklabels([])
            if j != len(rep) - 1:
                ax.set_xticklabels([])

            # Add repressor copy number to right plots
            if i == len(inducer) - 1:
                # Generate twin axis
                axtwin = ax.twinx()
                # Remove ticks
                axtwin.get_yaxis().set_ticks([])
                # Set label
                axtwin.set_ylabel(
                    r"rep. / cell = {:d}".format(r),
                    bbox=dict(facecolor="#ffedce"),
                )
                # Remove residual ticks from the original left axis
                ax.tick_params(color="w", width=0)

            if (j == 1) and (i == len(inducer) - 1):
                text = ax.text(
                    1.35,
                    0.5,
           r"$\Delta\epsilon_r = {:.1f} \; k_BT$".format(energies[k]),
                    size=10,
                    verticalalignment="center",
                    rotation=90,
                    color="white",
                    transform=ax.transAxes,
                    bbox=dict(facecolor=colors[0]),
                )

plt.savefig(figdir + "ECDF_theory_vs_data_regulated.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_53_0.png)


## Working at the mRNA level

Let's now work at the mRNA level. Just as for the protein, the mRNA inferences are done on a separate script at `./scripts/maxent_mRNA_dist.py`.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read resulting values for the multipliers.
df_maxEnt_mRNA = pd.read_csv(
    f"{datadir}MaxEnt_Lagrange_mult_mRNA.csv"
)
df_maxEnt_mRNA.head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>inducer_uM</th>
      <th>lambda_m1p0</th>
      <th>lambda_m2p0</th>
      <th>lambda_m3p0</th>
      <th>m0p1</th>
      <th>m0p2</th>
      <th>m0p3</th>
      <th>...</th>
      <th>m2p4</th>
      <th>m3p1</th>
      <th>m3p2</th>
      <th>m3p3</th>
      <th>m4p0</th>
      <th>m4p1</th>
      <th>m4p2</th>
      <th>m5p0</th>
      <th>m5p1</th>
      <th>m6p0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.149243</td>
      <td>-0.006137</td>
      <td>0.000026</td>
      <td>7732.570561</td>
      <td>6.240027e+07</td>
      <td>5.258396e+11</td>
      <td>...</td>
      <td>2.324187e+18</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.149243</td>
      <td>-0.006137</td>
      <td>0.000026</td>
      <td>7732.570561</td>
      <td>6.240027e+07</td>
      <td>5.258396e+11</td>
      <td>...</td>
      <td>2.324187e+18</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.149243</td>
      <td>-0.006137</td>
      <td>0.000026</td>
      <td>7732.570561</td>
      <td>6.240027e+07</td>
      <td>5.258396e+11</td>
      <td>...</td>
      <td>2.324187e+18</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.149243</td>
      <td>-0.006137</td>
      <td>0.000026</td>
      <td>7732.570561</td>
      <td>6.240027e+07</td>
      <td>5.258396e+11</td>
      <td>...</td>
      <td>2.324187e+18</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.149243</td>
      <td>-0.006137</td>
      <td>0.000026</td>
      <td>7732.570561</td>
      <td>6.240027e+07</td>
      <td>5.258396e+11</td>
      <td>...</td>
      <td>2.324187e+18</td>
      <td>7.866644e+07</td>
      <td>7.435490e+11</td>
      <td>7.223464e+15</td>
      <td>278504.362303</td>
      <td>2.639465e+09</td>
      <td>2.564953e+13</td>
      <td>1.054893e+07</td>
      <td>1.023094e+11</td>
      <td>4.526111e+08</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



Having determined the Lagrange multipliers let's look at the joyplots for the mRNA distribution.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract protein moments in constraints
mRNA_mom =  [x for x in df_maxEnt_mRNA.columns if 'lambda_' in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r'\d+', s))) for s in mRNA_mom]

# Define operators to be included
operators = ["O1", "O2", "O3"]

# Define repressors to be included
repressors = [22, 260, 1740]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt_mRNA.inducer_uM.unique())[::2]

# Define color for operators
# Generate list of colors
col_list = ["Blues", "Oranges", "Greens"]
col_dict = dict(zip(operators, col_list))

# Define binstep for plot, meaning how often to plot
# an entry
binstep = 1

# Define sample space
mRNA_space = np.arange(0, 50)
protein_space = np.array([0])

# Initialize plot
fig, ax = plt.subplots(
    len(repressors), len(operators), figsize=(5, 5), sharex=True, sharey=True
)

# Define displacement
displacement = 3e-2

# Loop through operators
for j, op in enumerate(operators):
    # Loop through repressors
    for i, rep in enumerate(repressors):

        # Extract the multipliers for a specific strain
        df_sample = df_maxEnt_mRNA[
            (df_maxEnt_mRNA.operator == op)
            & (df_maxEnt_mRNA.repressor == rep)
            & (df_maxEnt_mRNA.inducer_uM.isin(inducer))
        ]

        # Group multipliers by inducer concentration
        df_group = df_sample.groupby("inducer_uM", sort=True)

        # Extract and invert groups to start from higher to lower
        groups = np.flip([group for group, data in df_group])

        # Define colors for plot
        colors = sns.color_palette(col_dict[op], n_colors=len(df_group) + 1)

        # Initialize matrix to save probability distributions
        Pm = np.zeros([len(df_group), len(mRNA_space)])

        # Loop through each of the entries
        for k, group in enumerate(groups):
            data = df_group.get_group(group)

            # Select the Lagrange multipliers
            lagrange_sample = data.loc[
                :, [col for col in data.columns if "lambda" in col]
            ].values[0]

            # Compute distribution from Lagrange multipliers values
            Pm[k, :] = ccutils.maxent.maxEnt_from_lagrange(
                mRNA_space, protein_space, lagrange_sample, exponents=moments
            )

            # Generate PMF plot
            ax[i, j].plot(
                mRNA_space[0::binstep],
                Pm[k, 0::binstep] + k * displacement,
                drawstyle="steps",
                lw=1,
                color="k",
                zorder=len(df_group) * 2 - (2 * k),
            )
            # Fill between each histogram
            ax[i, j].fill_between(
                mRNA_space[0::binstep],
                Pm[k, 0::binstep] + k * displacement,
                [displacement * k] * len(mRNA_space[0::binstep]),
                color=colors[k],
                alpha=1,
                step="pre",
                zorder=len(df_group) * 2 - (2 * k + 1),
            )

        # Add x label to lower plots
        if i == 2:
            ax[i, j].set_xlabel("mRNA / cell")

        # Add y label to left plots
        if j == 0:
            ax[i, j].set_ylabel(r"[IPTG] ($\mu$M)")

        # Add operator top of colums
        if i == 0:
            label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(
                df_sample.binding_energy.unique()[0]
            )
            ax[i, j].set_title(label, bbox=dict(facecolor="#ffedce"))

        # Add repressor copy number to right plots
        if j == 2:
            # Generate twin axis
            axtwin = ax[i, j].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(
                r"rep. / cell = {:d}".format(rep),
                bbox=dict(facecolor="#ffedce"),
            )
            # Remove residual ticks from the original left axis
            ax[i, j].tick_params(color="w", width=0)

# Change ylim
ax[0, 0].set_ylim([-3e-3, 6e-2 + len(df_group) * displacement])
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.05, wspace=0.02)

# Set y axis ticks
yticks = np.arange(len(df_group)) * displacement
yticklabels = [int(x) for x in df_group.groups.keys()]

ax[0, 0].yaxis.set_ticks(yticks)
ax[0, 0].yaxis.set_ticklabels(yticklabels)

# Set x ticks every 10 mRNA
ax[0, 0].xaxis.set_ticks(np.arange(0, 50, 10))

# Save figure
plt.savefig(figdir + "PMF_grid_joyplot_mRNA.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_58_0.png)


## Extending the range of repressors.

The script `maxent_protein_dist_rep_range.py` repeats the inference of maximum entropy distributions for a larger number of repressor (up to $10^4$ repressors per cell). In what follows we will show that there is an optimal combination of operator-repressor copy number to maximize the amount of informatin that cells can process with this simple genetic circuit.

## Extending the range of inducer concentrations.

Also to test the consequences of limiting ourselves to 12 experimental inducer concentrations the script `maxent_protein_dist_iptg_range.py` performs the inference of the maximum entropy distributions for a finer grid of inducer values. Later on we will show that since the information processing capacity of the cells is of order $\approx 1.5$ using these smaller number of inducer concentrations suffices to capture the general trend.

## Testing how the number of moments included as constraints affects distribution predictions.

So far our analysis has included six moments of the protein distribution as constraints to construct the maximum entropy approximation. A valid question is how would these inferences be affected if less moments were included into the inference. For this we repeated inferences with varying number of constraints. These calculations are done in the script `maxent_protein_dist_var_mom.py`.

Let's read the resulting inferred maximum entropy distributions.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(f"{datadir}MaxEnt_Lagrange_mult_protein_var_mom.csv")
df_maxEnt.head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>inducer_uM</th>
      <th>num_mom</th>
      <th>lambda_m0p1</th>
      <th>lambda_m0p2</th>
      <th>lambda_m0p3</th>
      <th>lambda_m0p4</th>
      <th>lambda_m0p5</th>
      <th>lambda_m0p6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.003039</td>
      <td>-1.955020e-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>2</td>
      <td>0.003039</td>
      <td>-1.955020e-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2</td>
      <td>0.003039</td>
      <td>-1.955020e-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>2</td>
      <td>0.003039</td>
      <td>-1.955020e-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>2</td>
      <td>0.003039</td>
      <td>-1.955020e-07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Computing the KL divergence as a way to compare distributions

In order to quantify how much do distributions change as we change the number of constraints included for the inference we will compute the Kullback-Leibler (KL) divergence $D_{KL}$ defined as

$$
D_{KL}(P || Q) = \sum_x P(x) \log_2 \left( {P(x) \over Q(x)} \right),
\tag{27}
$$
where $P$ is the reference distribution with which $Q$ is comapared against. This quantity can be interpreted as the distance between two distributions, with the caveat that is a non-symmetric metric, i.e.

$$
D_{KL}(P || Q) \neq D_{KL}(Q || P).
\tag{28}
$$
An alternative interpretation of this metric is to think of it as the amount of information (in bits) lost when using $Q$ to represent the "real" distribution $P$. For our case we know that the more moment constraints included in the MaxEnt inference the more accurate it gets to the true distribution. Therefore we will compare all of our inferences with the reference distribution that includes the maximum number of constraints.

Let's compute these KL divergences.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Group by operator, repressor copy number 
# and inducer concentartion
df_group = df_maxEnt.groupby(['operator', 'binding_energy',
                              'repressor', 'inducer_uM'])

# Define names for columns in DataFrame to save KL divergences
names = ['operator', 'binding_energy', 'repressor', 
         'inducer_uM', 'num_mom', 'DKL', 'entropy']

# Initialize data frame to save KL divergences
df_kl = pd.DataFrame(columns=names)

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 4E4)

# Extract protein moments in constraints
prot_mom =  [x for x in df_maxEnt.columns if 'm0' in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r'\d+', s))) for s in prot_mom]

# Loop through groups
for group, data in df_group:
    # Extract parameters
    op = group[0]
    eR = group[1]
    rep = group[2]
    inducer = group[3]
    
    # List different number of moments
    num_mom = data.num_mom.unique()
    
    # Initialize matrix to save probability distributions
    Pp = np.zeros([len(num_mom), len(protein_space)])
    
    # Loop through number of moments
    for i, n in enumerate(num_mom):
        # Extract the multipliers 
        df_sample = df_maxEnt[(df_maxEnt.operator == op) &
                              (df_maxEnt.repressor == rep) &
                              (df_maxEnt.inducer_uM == inducer) &
                              (df_maxEnt.num_mom == n)]
        
        # Select the Lagrange multipliers
        lagrange_sample =  df_sample.loc[:, [col for col in data.columns 
                                         if 'lambda' in col]].values[0][0:n]

        # Compute distribution from Lagrange multipliers values
        Pp[i, :] = ccutils.maxent.maxEnt_from_lagrange(mRNA_space, 
                                                       protein_space, 
                                                       lagrange_sample,
                                                    exponents=moments[0:n]).T
        
    # Define reference distriution
    Pp_ref = Pp[-1, :]
    # Loop through distributions computing the KL divergence at each step
    for i, n in enumerate(num_mom):
        DKL = sp.stats.entropy(Pp_ref, Pp[i, :], base=2)
        entropy = sp.stats.entropy(Pp[i, :], base=2)
        
        # Generate series to append to dataframe
        series = pd.Series([op, eR, rep, inducer, 
                            n, DKL, entropy], index=names)
        
        # Append value to dataframe
        df_kl = df_kl.append(series, ignore_index=True)
        
df_kl.head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>inducer_uM</th>
      <th>num_mom</th>
      <th>DKL</th>
      <th>entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>0.000706</td>
      <td>12.690241</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0.000046</td>
      <td>12.708369</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>0.000008</td>
      <td>12.703792</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>0.000151</td>
      <td>12.711013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.000000</td>
      <td>12.705865</td>
    </tr>
  </tbody>
</table>
</div>



Let's take a look at the KL divergence with respect to the distribution with the most constraints.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Group data by operator
df_group = df_kl.groupby('operator')

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5),
                       sharex=True, sharey=True)

# Define colors for operators
col_list = ['Blues_r', 'Oranges_r', 'Greens_r']
col_dict = dict(zip(('O1', 'O2', 'O3'), col_list))

# Loop through operators
for i, (group, data) in enumerate(df_group):
    # Group by repressor copy number
    data_group = data.groupby('repressor')
    # Generate list of colors
    colors = sns.color_palette(col_dict[group], n_colors=len(data_group) + 1)
    
    # Loop through repressor copy numbers
    for j, (g, d) in enumerate(data_group):
        # Plot DK divergence vs number of moments
        ax[i].plot(d.num_mom, d.DKL, color=colors[j],
                   lw=0, marker='.', label=str(int(g)))
    
    # Change scale of y axis
    ax[i].set_yscale('symlog', linthreshy=1E-6)

    # Set y axis label
    ax[i].set_xlabel('number of moments')
    # Set title
    label = r'$\Delta\epsilon_r$ = {:.1f} $k_BT$'.\
               format(data.binding_energy.unique()[0])
    ax[i].set_title(label, bbox=dict(facecolor='#ffedce'))
    # Add legend
    ax[i].legend(loc='upper right', title='rep./cell', ncol=2,
                 fontsize=6)
    
# Set x axis label
ax[0].set_ylabel('KL divergenge (bits)')

# Adjust spacing between plots
plt.subplots_adjust(wspace=0.05)

# Save figure
plt.savefig(figdir + 'num_moments_vs_KL_div.pdf', 
            bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_70_0.png)


We can see that even using only the first two moments as constraints we already have a very small KL divergence. Recall that the KL divergence can be interpreted as the amount of information lost by assuming the wrong distribution. This means that even if we build the MaxEnt distribution using only the first two moments we only lose at most 0.2 bits of information compared with including all 6 constraints.

## Computing the single-promoter MaxEnt distributions.

To complete the comparison between the multi-promoter model with the single-promoter model we will now perform the MaxEnt inferences using the moment values obtained by computing the steady state moments of a single promoter for which $\gp > 0$, i.e. the protein degradation is assumed to be a Poisson process. We have previously shown that this model undersestimates the noise in gene expression (std/mean). But nevertheless it is still interesting to compare the MaxEnt inferences obtained with both models.

Let's first import the constraints.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Load moments for multi-promoter level
df_constraints_single = pd.read_csv('../../data/csv_maxEnt_dist/' + 
                        'MaxEnt_single_prom_constraints.csv')
```

### Inferences at the protein level for a single promoter

Let's start by computing the maximum entropy distributions at the protein level. First we define the sample space for the distributions.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract protein moments in constraints
prot_mom =  [x for x in df_constraints_single.columns if 'm0' in x]
# Define index of moments to be used in the computation
moments = [tuple(map(int, re.findall(r'\d+', s))) for s in prot_mom]

# Define sample space
mRNA_space = np.array([0])  # Dummy space
protein_space = np.arange(0, 10E4)

# Generate sample space as a list of pairs using itertools.
samplespace = list(itertools.product(mRNA_space, protein_space))

# Initialize matrix to save all the features that are fed to the
# maxentropy function
features = np.zeros([len(moments), len(samplespace)])

# Loop through constraints and compute features
for i, mom in enumerate(moments):
    features[i, :] = [ccutils.maxent.feature_fn(x, mom) for x in samplespace]
```

Now we are ready to perform the MaxEnt inferences


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Indicate if the computation should be performed
fit_dist = False

# Initialize data frame to save the lagrange multipliers.
names = ['operator', 'binding_energy', 'repressor', 'inducer_uM']
# Add names of the constraints
names = names + ['lambda_m' + str(m[0]) + 'p' + str(m[1]) for m in moments]

# Initialize empty dataframe
df_maxEnt = pd.DataFrame([], columns=names)

# Define column names containing the constraints used to fit the distribution
constraints_names = ['m' + str(m[0]) + 'p' + str(m[1]) for m in moments]

if fit_dist:
    # Define function for parallel computation
    def maxEnt_parallel(idx, df):
        # Report on progress
        print('iteration: ',idx)
            
        # Extract constraints
        constraints = df.loc[constraints_names]
        
        # Perform MaxEnt computation
        # We use the Powell method because despite being slower it is more
        # robust than the other implementations.
        Lagrange = MaxEnt_bretthorst(constraints, features, 
                                     algorithm='Powell', 
                                     tol=1E-5, paramtol=1E-5,
                                     maxiter=10000)
        # Save Lagrange multipliers into dataframe
        series = pd.Series(Lagrange, index=names[4::])
        
        # Add other features to series before appending to dataframe
        series = pd.concat([df.drop(constraints_names), series])
        
        return series
    
    # Run the function in parallel
    maxEnt_series = Parallel(n_jobs=6)(delayed(maxEnt_parallel)(idx, df)
                           for idx, df in df_constraints_single.iterrows())
    
    # Initialize data frame to save list of parameters
    df_maxEnt = pd.DataFrame([], columns=names)

    for s in maxEnt_series:
        df_maxEnt = df_maxEnt.append(s, ignore_index=True)

    df_maxEnt.to_csv(datadir + 'MaxEnt_Lagrange_single_protein.csv',
                     index=False)
    
# Read resulting values for the multipliers.
df_maxEnt = pd.read_csv(datadir + 'MaxEnt_Lagrange_single_protein.csv')
df_maxEnt.head()
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>inducer_uM</th>
      <th>lambda_m0p1</th>
      <th>lambda_m0p2</th>
      <th>lambda_m0p3</th>
      <th>m1p0</th>
      <th>m2p0</th>
      <th>m3p0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014171</td>
      <td>-7.160163e-07</td>
      <td>-2.897475e-12</td>
      <td>18.720041</td>
      <td>432.6068</td>
      <td>11706.453789</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>0.014171</td>
      <td>-7.160163e-07</td>
      <td>-2.897475e-12</td>
      <td>18.720041</td>
      <td>432.6068</td>
      <td>11706.453789</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.014171</td>
      <td>-7.160163e-07</td>
      <td>-2.897475e-12</td>
      <td>18.720041</td>
      <td>432.6068</td>
      <td>11706.453789</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.014171</td>
      <td>-7.160163e-07</td>
      <td>-2.897475e-12</td>
      <td>18.720041</td>
      <td>432.6068</td>
      <td>11706.453789</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.014171</td>
      <td>-7.160163e-07</td>
      <td>-2.897475e-12</td>
      <td>18.720041</td>
      <td>432.6068</td>
      <td>11706.453789</td>
    </tr>
  </tbody>
</table>
</div>



Let's take a look at some of these distributions.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define operators to be included
operators = ['O1', 'O2', 'O3']

# Define repressors to be included
repressors = [22, 260, 1740]

# Define concnentration to include in plot
inducer = np.sort(df_maxEnt.inducer_uM.unique())[::2]

# Define color for operators
# Generate list of colors
col_list = ['Blues', 'Oranges', 'Greens']
col_dict = dict(zip(operators, col_list))

# Define binstep for plot, meaning how often to plot
# an entry
binstep = 100

# Define sample space
mRNA_space = np.array([0])
protein_space = np.arange(0, 1.8E4)

# Initialize plot
fig, ax = plt.subplots(len(repressors), len(operators), figsize=(5, 5),
                       sharex=True, sharey=True)

# Define displacement
displacement = 5E-5

# Loop through operators
for j, op in enumerate(operators):
    # Loop through repressors
    for i, rep in enumerate(repressors):

        # Extract the multipliers for a specific strain
        df_sample = df_maxEnt[(df_maxEnt.operator == op) &
                              (df_maxEnt.repressor == rep) &
                              (df_maxEnt.inducer_uM.isin(inducer))]

        # Group multipliers by inducer concentration
        df_group = df_sample.groupby('inducer_uM', sort=True)

        # Extract and invert groups to start from higher to lower
        groups = np.flip([group for group, data in df_group])
        
        # Define colors for plot
        colors = sns.color_palette(col_dict[op], n_colors=len(df_group)+1)
        
        # Initialize matrix to save probability distributions
        Pp = np.zeros([len(df_group), len(protein_space)])

        # Loop through each of the entries
        for k, group in enumerate(groups):
            data = df_group.get_group(group)
            
            # Select the Lagrange multipliers
            lagrange_sample =  data.loc[:, [col for col in data.columns 
                                                 if 'lambda' in col]].values[0]

            # Compute distribution from Lagrange multipliers values
            Pp[k, :] = ccutils.maxent.maxEnt_from_lagrange(mRNA_space, 
                                                           protein_space, 
                                                           lagrange_sample,
                                                           exponents=moments).T

            # Generate PMF plot
            ax[i, j].plot(protein_space[0::binstep], Pp[k, 0::binstep] + k * displacement,
                          drawstyle='steps', lw=1,
                          color='k', zorder=len(df_group) * 2 - (2 * k))
            # Fill between each histogram
            ax[i, j].fill_between(protein_space[0::binstep], Pp[k, 0::binstep] + k * displacement,
                                  [displacement * k] * len(protein_space[0::binstep]), 
                                  color=colors[k], alpha=1, step='pre', 
                                  zorder=len(df_group) * 2 - (2 * k + 1))

        # Add x label to lower plots
        if i==2:
            ax[i, j].set_xlabel('protein / cell')   
            
        # Add y label to left plots
        if j==0:
            ax[i, j].set_ylabel('[IPTG] ($\mu$M)')
        
        # Add operator top of colums
        if i==0:
            label = r'$\Delta\epsilon_r$ = {:.1f} $k_BT$'.\
                    format(df_sample.binding_energy.unique()[0])
            ax[i, j].set_title(label, bbox=dict(facecolor='#ffedce'))
            
        # Add repressor copy number to right plots
        if j==2:
            # Generate twin axis
            axtwin = ax[i, j].twinx()
            # Remove ticks
            axtwin.get_yaxis().set_ticks([])
            # Set label
            axtwin.set_ylabel(r'rep. / cell = {:d}'.format(rep),
                              bbox=dict(facecolor='#ffedce'))
            # Remove residual ticks from the original left axis
            ax[i, j].tick_params(color='w', width=0)

# Change lim
ax[0, 0].set_ylim([-3E-5, 6.5E-4 + len(df_group) * displacement])
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.02, wspace=0.02)

# Set y axis ticks
yticks = np.arange(len(df_group)) * displacement
yticklabels = [int(x) for x in groups]

ax[0, 0].yaxis.set_ticks(yticks)
ax[0, 0].yaxis.set_ticklabels(yticklabels)

# Set x axis ticks
xticks = [0, 5E3, 1E4, 1.5E4]
ax[0, 0].xaxis.set_ticks(xticks)

# Save figure
plt.savefig(figdir + 'PMF_grid_joyplot_protein_single.pdf',
            bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](MaxEnt_approx_joint_files/MaxEnt_approx_joint_81_0.png)


These distributions are definitely much less noisy compared with the multi-promoter model.
