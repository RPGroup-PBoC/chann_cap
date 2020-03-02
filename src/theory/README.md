# Theory related code

In this folder there is all of the code used for computations related to the
master-equation representation of the simple-repression motif. Everything from
the analytical computation of moments of a distribution, to the computation of
the channel capacity of the simple genetic circuit. Each of the `jupyter`
notebooks is highly annotated to explain each and every step of the process.
Here we just give a general overview of each of the files in this folder.

#### `chemical_master_mRNA_FISH_mcmc.ipynb`
In this notebook we use the closed-form solution of the two-state promoter mRNA
distribution to infer the kinetic parameters related to the unregulated
promoter using Markov Chain Monte Carlo. More specifically we take the original
data from [Jones, Brewster & Phillips, 2014](https://science.sciencemag.org/content/346/6216/1533)
and fit the parameters $k^{(p)}_\text{on}$, $k^{(p)}_\text{off}$, and $r_p$.

#### `chemical_master_steady_state_moments_general.ipynb`
In this notebook we analytically derive the form of the mRNA and protein
distribution assuming that the dynamics reach steady state. This computations
is possible because, as explained in the text, the master equation has moment
closure, meaning that high moments of the distribution can be computed by
knowing lower moments. The output of this notebook are `lambdify` functions
that allow the numerical substitution of the rate parameters into the
analytical results obtained with `sympy`.

#### `moment_dynamics_system.ipynb`
This notebook generates the right-hand side of the ODEs for the mRNA and
protein moment distributions. Instead of assuming steady-state as the
`chemical_master_steady_state_moments_general.ipynb` notebook, here we just
define how the moments of the distribution evolve over time. The output of this
notebook can then be used with `scipy`'s `odeint` function to numerically
integrate these dynamics.

#### `binomial_moments.ipynb`
In this notebook we compute what the moments of the mRNA and protein
distribution should look like right after cell division when each of these
molecules undergoes a binomial partitioning. For example, the mean mRNA copy
number in a cell right after it divides $\left\langle m \right\rangle_{t_d}$
should exactly be **half of the mean mRNA count before cell division** since on
average each daughter cell will end up with half the amount of molecules as the
mother cell. In this notebook we therefore infer that the coefficient to
compute the first moment of the mRNA distribution after cell division is
**1/2** the first moment of the mRNA distribution before cell division. For
other moments such coefficients are not as obvious, so in the notebook we use
`sympy` to obtain such quantities.

#### `moment_dynamics_cell_division.ipynb`
In this notebook we use the outcome from `moment_dynamics_system.ipynb` in
combination with the outcome from `binomial_moments.ipynb` to compute the time
evolution of the moments of the mRNA and protein distribution as cells progress
through the cell cycle. This is because as cells grow they replicate their
genome, spending a fraction of the cell cycle with two rather than a single
copy of a gene. This, along with the binomial partitioning of the molecules
among daughter cells after cell division, is an important source of
cell-to-cell variability that need to be accounted for.

#### `MaxEnt_approx_joint.ipynb`
This notebook implements the maximum entropy approach to approximate the joint
mRNA and protein distribution given information only about the moments of such
distribution. The computational task here consists in finding the numerical
value of the Lagrange multipliers associated with each of the moments to
approximate the full distribution.

#### `blahut_algorithm_channel_capacity.ipynb`
In this notebook we implement the Blahut-Arimoto algorithm to compute the
channel capacity given an input-output function. This allows us to compute the
maximum amount of information that a simple-repression motif can process.

#### `gillespie_simulation.ipynb`
This notebook generates stochastic simulations of the two-state unregulated
promoter over several cell cycles using the Gillespie algorithm. We use this to
validate our maximum entropy approximation of the mRNA and protein distribution.
