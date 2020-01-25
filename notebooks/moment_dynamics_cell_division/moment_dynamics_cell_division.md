# Moment dynamics with cell division

(c) 2020 Manuel Razo. This work is licensed under a [Creative Commons Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). All code contained herein is licensed under an [MIT license](https://opensource.org/licenses/MIT)

---


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
import os
import pickle
import cloudpickle
import itertools
import glob

# Our numerical workhorses
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm

# Import libraries to parallelize processes
from joblib import Parallel, delayed

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Seaborn, useful for graphics
import seaborn as sns

# Import the project utils
import ccutils

# Magic function to make matplotlib inline; other style specs must come AFTER
%matplotlib inline

# This enables SVG graphics inline
%config InlineBackend.figure_format = 'retina'

tmpdir = '../../tmp/'
figdir = '../../fig/moment_dynamics_numeric/'
datadir = '../../data/csv_maxEnt_dist/'
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

$$\newcommand{kpon}{k^{(p)}_{\text{on}}}$$
$$\newcommand{kpoff}{k^{(p)}_{\text{off}}}$$
$$\newcommand{kron}{k^{(r)}_{\text{on}}}$$
$$\newcommand{kroff}{k^{(r)}_{\text{off}}}$$
$$\newcommand{rm}{r _m}$$
$$\newcommand{gm}{\gamma _m}$$
$$\newcommand{rp}{r _p}$$
$$\newcommand{gp}{\gamma _p}$$
$$\newcommand{mm}{\left\langle m \right\rangle}$$
$$\newcommand{ee}[1]{\left\langle #1 \right\rangle}$$
$$\newcommand{bb}[1]{\mathbf{#1}}$$
$$\newcommand{foldchange}{\text{fold-change}}$$
$$\newcommand{\ee}[1]{\left\langle #1 \right\rangle}$$
$$\newcommand{\bb}[1]{\mathbf{#1}}$$
$$\newcommand{\dt}[1]{ {\partial{#1} \over \partial t}}$$
$$\newcommand{\Km}{\bb{K}}$$
$$\newcommand{\Rm}{\bb{R}_m}$$
$$\newcommand{\Gm}{\bb{\Gamma}_m}$$
$$\newcommand{\Rp}{\bb{R}_p}$$
$$\newcommand{\Gp}{\bb{\Gamma}_p}$$

## Distribution moment dynamics with cell division 

As first discussed by [Jones et al.](http://science.sciencemag.org/content/346/6216/1533) and then further expanded by [Peterson et al.](http://www.pnas.org/content/112/52/15886) the effect of having multiple gene copy numbers due to genome replication during the cell cycle has an important effect on gene expression noise. As the genome is replicated the cells spend part of their cell cycle with > 1 copy of the gene. The number of copies depends on the growth rate and on the gene position relative to the genome replication origin.

For our experimental setup our cells spend 40% of the cell cycle with 2 copies of the reporter gene and 60% with one copy. We previously inferred the parameters $\kpon$, $\kpoff$, and $r_m$ assuming that at both stages the mRNA reached steady state with $r_m$ as production rate for 1 gene copy and $2 r_m$ for two copies. The objective of this notebook is to explore the dynamical consequences of these gene copy number variations at the level of the mRNA and protein distribution moments.

The first thing we need to establish are the dynamics for the mRNA. The cell doubling time $t_d = 90$ min establishes the period of the cell cycle. For a time $0 < t_s < t_d$ cells have 1 copy of the gene (i.e. mRNA production rate $r_m$) and for the rest of the cycle the cells have 2 copies of the gene (i.e. mRNA production rate of $2 r_m$).

Therefore for our simulations we will initialize the moments at the steady state values for the single promoter, run the simulation with those parameters for time 0 to $t_s$ and then change the parameters for the rest of the simulation until reaching time $t_d$.

## Cell division and bionomial partitioning of molecules

On the notebook `binomial_moments.ipynb` we show that the moments after the cell division can be computed analytically as a linear combination of the moments before the cell division. For this we created a matrix $\bb{Z}$ that contains the coefficients of this linear combination. Let's read the matrix into memory


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read matrix into memory
with open('./pkl_files/binom_coeff_matrix.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    Z_mat = unpickler.load()
    expo_binom = unpickler.load()
```

## Protein degradation as a non-Poission process

As written in the master equation the protein degradation is also a Poisson process with rate $\gp$. But the way this rate was determined is by establishing that the main source of protein degradation comes from dilution during cell growth. These two statements contractic each other. Since we will be working with the explicit dynamics during the cell cycle, we will set the protein degradation rate to $\gp = 0$, having the protein degradation come only from the dilution as cells divide.

Let's begin by defining the promoter parameters. For now we will not define protein production rate $r_p$. Later on we will come back to this parameter setting its value value to satisfy what is know about the mean protein copy number per mRNA.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Load constants
param = ccutils.model.load_constants()

# Define protein production and degradatino rates
gp = 0 # sec^-1
```

## Defining moment dynamics 

We have already on the `moment_dynamics_system.ipynb` notebook established the dynamics up to the 6th protein moment. In general we established that the moment dynamics are of the form

$$
\dt{\bb{\mu^{(x, y)}}} = \bb{A \mu^{(x, y)}},
\tag{1}
$$
where $\bb{\mu^{(x, y)}}$ is the array containing all of our moments, and the matrix $\bb{A}$ contains the linear coefficients of our linear system.

Let us begin by defining a function `dmomdt` that takes as input an array of moments `m`, a time array `t` and a matrix `A` and returns the right-hand side of the equation for the moment dynamics. This function will be fed to the `scipy.integrate.odeint` function.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
def dmomdt(mom, t, A):
    '''
    Function that computes the right-hand side of the moment
    dynamics equation
    dµ/dt = Aµ
    This funciton is fed to the scipy.integrate.odeint function
    Parameters
    ----------
    mom : array-like
        Array containing all of the moments included in the matrix
        dynamics A.
    t : array-like
        time array
    A : 2D-array.
        Matrix containing the linear coefficients of the moment
        dynamics equation
    Returns
    -------
    Right hand-side of the moment dynamics
    '''
    return np.dot(A, mom)
```

### Running dynamics until steady state.

If our model were not to consider explicit cell divisions, and we were to set the protein degradation rate to be $\gp > 0$ all moments of the distribution would reach a steady state-value. Taking advantage of that fact we will use these steady-state values as the initial condition for our numerical integration. This is because starting at a non-zero value that is close to the value that the moments would experience over cell cycles makes more sense than starting all moments for example at zero.

Let's define the matrix $\bb{A}$ to compute the initial conditions then. Again, since we will not explicitly include cell divisions we need to set a non-zero degradation rate such that these steady state value can be reached. Therefore to compute these initial conditions we will use a production rate `rp_init` and a protein degradation rate `gp_init` such that the mean protein copy number is 500 times the mean mRNA copy number. that means that ${r_p \over \gp} = 500$.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Single promoter
gp_init = 1 / (60 * 60)
rp_init = 500 * gp_init

# Read protein ununregulated matrix 
with open('./pkl_files/two_state_protein_dynamics_matrix.pkl', 'rb') as file:
    # Load sympy object containing the matrix A that define the
    # moment dynamics
    A_mat_unreg_lam = cloudpickle.load(file)
    # Load the list of moments included in the matrix
    expo = cloudpickle.load(file)
    
# Substitute value of parameters on matrix
##  Initial conditions
A_mat_unreg_s_init = A_mat_unreg_lam(param['kp_on'], param['kp_off'],
                                     param['rm'], param['gm'],
                                     rp_init, gp_init)
```

Now we will run the dynamics with the "artificial" $r_p$ and $\gp$ for a long time (equivalent to several cell cycles) such that all moments reach steady state.
We will initialize all moments except the zeroth moment to be zero. The zeroth moment represents the probability of being on any of the promoter states, and the sum has to always add up to 1. So given this subtle but important detail let's run the dynamics until reaching steady state.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define time on which to perform integration
t = np.linspace(0, 4000 * 60, 2000)

# Define initial conditions
mom_init = np.zeros(len(expo) * 2)
# Set initial condition for zero moment
# Since this needs to add up to 1
mom_init[0] = 1

# Numerically integrate equations
mp_sol = sp.integrate.odeint(dmomdt, mom_init, t, 
                             args=(A_mat_unreg_s_init,))

mp_init = mp_sol[-1, :]

print('<m> = {:.1f}'.format(mp_init[2:4].sum()))
print('<p> = {:.1f}'.format(mp_init[14:16].sum()))
print('<p>/<m> = {:.1f}'.format(mp_init[14:16].sum() / mp_init[2:4].sum()))
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    <m> = 12.0
    <p> = 5982.3
    <p>/<m> = 500.0


Excellent so we can see from this that as expected the mean protein copy number per mRNA is 500. That indicates that the dynamics ran long enough to reach the expected steady state.

Now we are in position to perform the correct integration over cell cycles with explicit cell divisions.

### Defining function to compute moments over severall cell cycles

Let's now define a function that computes the momeny dynamics over several cell cycles. Every time the cells divide we will use the matrix `Z_mat` to compute the moments after the cell division.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
def dmomdt_cycles(mom_init, t_single, t_double,
                  A_mat_fun, par_single, par_double,
                  expo, n_cycles, Z_mat,
                  n_steps=1000, states=['A', 'I']):
    '''
    Function that integrates the moment dynamics over several cell 
    cycles. The dynamics are integrated assuming a non-poisson
    protein degradation. So the protein is only degraded due to 
    cell division.
    
    Parameters
    ----------
    mom_init : array-like.
        Array containing the  initial conditions for the moment 
        of the states of the promoter.
    t_single : float.
        Time [in 1/mRNA degradation rate units] that cells spend 
        with a single promoter copy
    t_double : float.
        Time [in 1/mRNA degradation rate units] that cells spend 
        with a two promoter copies.
    A_mat_fun: function.
        Function to build the matrix moment dynamics. 
        This function takes as input the necessary rates 
        to build the matrix that defines the dynamics
        dµ/dt = A_mat * µ.
    par_single, par_double: list.
        Lists containing the rate parameters to be fed into the
        A_mat_fun function. These parameters must come in the 
        correct order that will be fed into the funciton.
        par_single = parameters for single promoter
        par_double = parameter for two promoters
    expo : array-like
        List containing the moments involved in the 
        dynamics defined by A
    n_cycles : int.
        Number of cell cycles to integrate for. A cell cycle is defined
        as t_single + t_double.
    Z_mat : array-like.
        Array containing the linear coefficients to compute the moments
        after the cell division
    n_steps : int. Default = 1000.
        Number of steps to use for the numerical integration.
    states : array-like. Default = ['A', 'I']
        Array containing the strings that define the moments that the
        promoter can be found at. For an unregulated promoter the only
        two available states are 'A' (active state) and 'E' (inactive).
        For the regulated case a third state 'R' (repressor bound) is
        available to the system.

    Returns
    -------
    distribution moment dynamics over cell cycles
    '''
    # Initialize names for moments in data frame
    names = ['m{0:d}p{1:d}'.format(*x) + s for x in expo 
             for s in states]
    
    # Substitute value of parameters on matrix
    # Single promoter
    A_mat_s = A_mat_fun(*par_single)
    # Two promoters
    A_mat_d = A_mat_fun(*par_double)

    # Generate division matrix for all states
    # Initialize matrix
    Z_mat_div = np.zeros([len(names), len(names)])
    
    # Loop through exponents
    for i, e in enumerate(expo):
        # Loop through states
        for j, s in enumerate(states):
            Z_mat_div[(i * len(states)) + j,
                      j::len(states)] = Z_mat[i]
    
    # Initialize data frame
    df = pd.DataFrame(columns=['time', 'state', 'cycle'] + names)
    
    # Initilaize global time
    t_sim = 0
    
    ###  Loop through cycles  ###
    for cyc in range(n_cycles):
        # == Single promoter == #
        # Define time array
        t = np.linspace(0, t_single, n_steps)

        # Integrate moment equations
        mom = sp.integrate.odeint(dmomdt, mom_init, t, 
                             args=(A_mat_s,))

        # Generate data frame
        df_mom = pd.DataFrame(mom, columns=names)
        # Append time, state and cycle
        df_mom = df_mom.assign(time=t + t_sim)
        df_mom = df_mom.assign(state=['single'] * mom.shape[0])
        df_mom = df_mom.assign(cycle=[cyc] * mom.shape[0])
        
        # Append results to global data frame
        df = df.append(df_mom, ignore_index=True, sort=False)
        
        # Update global time
        # NOTE: Here we account for whether or not this is the first cycle
        # This is because of the extra time bit we have to add in order not
        # to have two overlapping time points
        if cyc == 0:
            t_sim = t_sim + t[-1]
        else:
            t_sim = t_sim + t[-1] + np.diff(t)[0]
        
        # == Two promoters == #
        
        # Define initial conditions as last 
        # point of single promoter state
        mom_init = mom[-1, :]
        
        # Define time array
        t = np.linspace(0, t_double, n_steps)

        # Integrate moment equations
        mom = sp.integrate.odeint(dmomdt, mom_init, t, 
                                  args=(A_mat_d,))

        # Generate data frame
        df_mom = pd.DataFrame(mom, columns=names)
        # Append time, state and cycle
        df_mom = df_mom.assign(time=t + t_sim)
        df_mom = df_mom.assign(state=['double'] * mom.shape[0])
        df_mom = df_mom.assign(cycle=[cyc] * mom.shape[0])
        
        # Append results to global data frame
        df = df.append(df_mom, ignore_index=True, sort=False)
        
        # Update global time
        t_sim = t_sim + t[-1] + np.diff(t)[0]
        
        # == Cell division == #
        
        # Extract moments during last time point
        mom_fix = mom[-1, :]
        
        # Compute moments after cell division
        mom_init = np.dot(Z_mat_div, mom_fix)
        
    return df
```

Having defined these functions let's first test them with the two-state unregulated promoter. We already imported thte matrix $\bb{A}$ containing the coefficients for the dynamics, so we have everything we need. 

### Systematically choosing value for $r_p$

What we are missing is a proper value for the protein production rate $r_p$. In principle this parameter depends on the number of available ribosomes in the cell and the strenght of the ribosomal binding site on our reporter mRNA. But, as stated before, we know that on average there are 500 proteins per mRNA in cells. So let's set a function to find a rate $r_p$ that satisfies this condition. This will not be a complicated very general function, but a simple hard-coded routine to quickly get at a value of $r_p$.

First we define a function that computes the difference between the desired mean protein per mRNA ($\approx 500$) from what the actual value of $r_p$ gives. We will then use a minimization routine to minimize this residual as we change $r_p$.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
def rp_residual(rp, mp_init, p_m=500, param=param):
    '''
    Function used by the minimization routine to find the protein
    production rate that gives the desired protein to mRNA ratio.
    '''
    # Define doubling time
    doubling_time = 100
    # Define fraction of cell cycle spent with one copy
    t_single_frac = 0.6
    # Define time for single-promoter state
    t_single = 60 * t_single_frac * doubling_time # sec
    t_double = 60 * (1 - t_single_frac) * doubling_time # sec

    # Define number of cell cycles
    n_cycles = 6

    # Define list of parameters
    par_single = [param['kp_on'], param['kp_off'], param['rm'], param['gm'],
                  rp, 0]
    par_double = [param['kp_on'], param['kp_off'], 2 * param['rm'],
                  param['gm'], rp, 0]

    # Integrate moment equations
    df_p_unreg = dmomdt_cycles(mp_init, t_single, t_double, 
                               A_mat_unreg_lam, 
                               par_single, par_double, expo,
                               n_cycles, Z_mat, n_steps=2500)
    
    # Extract index for mRNA and protein first moment
    first_mom_names_m = [x for x in df_p_unreg.columns
                         if 'm1p0' in x]
    first_mom_names_p = [x for x in df_p_unreg.columns
                         if 'm0p1' in x]

    # Extract the last cycle information
    df_m_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      first_mom_names_m]
    df_p_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      first_mom_names_p]

    # Extract time of last cell cycle
    time = np.sort(df_p_unreg.loc[df_p_unreg.cycle == 
                                  df_p_unreg.cycle.max(),
                                  'time'].unique())
    
    # Integrate mean mRNA and mean protein using the cell age
    # distribution.
    # Compute the time differences
    time_diff = np.diff(time)

    # Compute the cumulative time difference
    time_cumsum = np.cumsum(time_diff)
    time_cumsum = time_cumsum / time_cumsum[-1]

    # Define array for spacing of cell cycle
    a_array = np.zeros(len(time))
    a_array[1:] = time_cumsum 
    
    # Compute probability based on this array
    p_a_array = np.log(2) * 2**(1 - a_array)

    # Perform numerical integration
    m_mean_unreg = sp.integrate.simps(df_m_unreg_first.sum(axis=1) *
                                      p_a_array, a_array)
    p_mean_unreg = sp.integrate.simps(df_p_unreg_first.sum(axis=1) * 
                                      p_a_array, a_array)
    
    return  np.abs(p_m - p_mean_unreg / m_mean_unreg)
```

Let's now find the protein production rate $r_p$.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Compute mean protein per mRNA
rp_opt = sp.optimize.minimize_scalar(rp_residual,
                                     bounds=(0, 0.1), method='bounded',
                                     args=(mp_init, 500))

rp = rp_opt.x
rp_opt
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



         fun: 0.008401790779544172
     message: 'Solution found.'
        nfev: 14
      status: 0
     success: True
           x: 0.05768706295740175



Excellent. Now that we have all parameters we are ready to run the dynamics for the unregulated promoter over several cell cycles! Let's do it to make sure that our value for the protein production rate indeed satisfies the desired ratio ${\ee{p}\over\ee{m} = 500$.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define doubling time
doubling_time = 100
# Define fraction of cell cycle spent with one copy
t_single_frac = 0.6
# Define time for single-promoter state
t_single = 60 * t_single_frac * doubling_time # sec
t_double = 60 * (1 - t_single_frac) * doubling_time # sec

# Define number of cell cycles
n_cycles = 6

# Set the protein production rate to the value obtained
# to give the right protein / mRNA ratio
rp = rp_opt.x

# Define list of parameters
par_single = [param['kp_on'], param['kp_off'], param['rm'], param['gm'],
              rp, 0]
par_double = [param['kp_on'], param['kp_off'], 2 * param['rm'],
              param['gm'], rp, 0]

# Integrate moment equations
df_p_unreg = dmomdt_cycles(mp_init, t_single, t_double, 
                           A_mat_unreg_lam, 
                           par_single, par_double, expo,
                           n_cycles, Z_mat, n_steps=10000)

# Extract index for mRNA and protein first moment
first_mom_names_m = [x for x in df_p_unreg.columns
                     if 'm1p0' in x]
first_mom_names_p = [x for x in df_p_unreg.columns
                     if 'm0p1' in x]

# Extract the last cycle information
df_m_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == df_p_unreg.cycle.max(),
                                  first_mom_names_m]
df_p_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == df_p_unreg.cycle.max(),
                                  first_mom_names_p]
# Extract time of last cell cycle
time = np.sort(df_p_unreg.loc[df_p_unreg.cycle == 
                              df_p_unreg.cycle.max(),
                              'time'].unique())

# Compute the time differences
time_diff = np.diff(time)

# Compute the cumulative time difference
time_cumsum = np.cumsum(time_diff)
time_cumsum = time_cumsum / time_cumsum[-1]

# Define array for spacing of cell cycle
a_array = np.zeros(len(time))
a_array[1:] = time_cumsum 

# Compute probability based on this array
p_a_array = np.log(2) * 2**(1 - a_array)

# Perform numerical integration
m_mean_unreg = sp.integrate.simps(df_m_unreg_first.sum(axis=1) * p_a_array,
                                  a_array)
p_mean_unreg = sp.integrate.simps(df_p_unreg_first.sum(axis=1) * p_a_array,
                                  a_array)

print('unregulated promoter:')
print('<m> = {:.2f}'.format(m_mean_unreg))
print('<p> = {:.2f}'.format(p_mean_unreg))
print('<p>/<m> = {:.1f}'.format(p_mean_unreg / m_mean_unreg))
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    unregulated promoter:
    <m> = 15.47
    <p> = 7732.54
    <p>/<m> = 500.0


### Plotting dynamics for unregualted promoter

Our protein production rate satisfies the expected condition. Now let's plot the dynamics over several cell cycles. As a summary we will display the dynamics as mean $\pm$ standard devaition for both the protein and mRNA. We acknowledge that the distributions might not be symmetric, therefore plotting symmetric standard deviations is not necessarily correct, but it is just to give intuition about our computation.

Let's first compute the variance.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract index for first moment
first_mom_names_m = [x for x in df_p_unreg.columns if 'm1p0' in x]
first_mom_names_p = [x for x in df_p_unreg.columns if 'm0p1' in x]

# Compute the mean mRNA copy number
m_mean = df_p_unreg.loc[:, first_mom_names_m].sum(axis=1)
p_mean = df_p_unreg.loc[:, first_mom_names_p].sum(axis=1)

# Extrac second moment
second_mom_names_m = [x for x in df_p_unreg.columns if 'm2p0' in x]
second_mom_names_p = [x for x in df_p_unreg.columns if 'm0p2' in x]

# Compute the second moments
m_second = df_p_unreg.loc[:, second_mom_names_m].sum(axis=1)
p_second = df_p_unreg.loc[:, second_mom_names_p].sum(axis=1)

# Compute variance
m_var = m_second - m_mean**2
p_var = p_second - p_mean**2
```

Now we are ready to plot the mean plus standard deviation.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define colors
colors = sns.color_palette('Paired', n_colors=2)

# Initialize figure
fig, ax = plt.subplots(2, 1, figsize=(2.5, 2), sharex=True)

# Plot mean as solid line
ax[0].plot(df_p_unreg.time / 60, m_mean, label='', lw=1.25,
           color=colors[1])
ax[1].plot(df_p_unreg.time / 60, p_mean, label='', lw=1.25,
           color=colors[1])

# Plot +- standard deviation 
ax[0].fill_between(df_p_unreg.time / 60, 
                   y1=m_mean + np.sqrt(m_var),
                   y2=m_mean - np.sqrt(m_var),
                   label='', color=colors[0], alpha=0.85,
                   zorder=2)
ax[1].fill_between(df_p_unreg.time / 60, 
                   y1=p_mean + np.sqrt(p_var),
                   y2=p_mean - np.sqrt(p_var),
                   label='', color=colors[0], alpha=0.85,
                   zorder=2)

# Group data frame by cell cycle
df_group = df_p_unreg.groupby('cycle')


# Loop through cycles
for i, (group, data) in enumerate(df_group):
    # Define the label only for the last cell cycle not to repeat in legend
    if group == df_p_unreg['cycle'].max():
        label_s = 'single promoter'
        label_d = 'two promoters'
    else:
        label_s = ''
        label_d = ''
    # Find index for one-promoter state
    idx = np.where(data.state == 'single')[0]
    # Indicate states with two promoters
    ax[0].axvspan(data.iloc[idx.min()]['time'] / 60, 
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#e3dcd1', label=label_s)
    ax[1].axvspan(data.iloc[idx.min()]['time'] / 60, 
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#e3dcd1', label='')
    
    # Find index for two-promoter state
    idx = np.where(data.state == 'double')[0]
    # Indicate states with two promoters
    ax[0].axvspan(data.iloc[idx.min()]['time'] / 60,
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#ffedce', label=label_d)
    ax[1].axvspan(data.iloc[idx.min()]['time'] / 60,
                  data.iloc[idx.max()]['time'] / 60,
               facecolor='#ffedce', label='')

##  Indicate where the cell divisions happen
# First find where the cell cycle transition happen
trans_idx = np.array(np.diff(df_p_unreg.cycle) == 1)
# Add extra point to have same length
trans_idx = np.insert(trans_idx, 0, False)  
# Get the time points at which this happens
time_div = df_p_unreg[trans_idx].time.values
# Plot with a triangle the cell division moment
ax[0].plot(time_div / 60, [np.max(m_mean) * 1.1] * len(time_div),
           lw=0, marker='v', color='k')

# Set limits
# mRNA
ax[0].set_xlim(df_p_unreg['time'].min() / 60, df_p_unreg['time'].max() / 60)
ax[0].set_ylim([0, 40])
#protein
ax[1].set_xlim(df_p_unreg['time'].min() / 60, df_p_unreg['time'].max() / 60)
ax[1].set_ylim([5000, 14000])

# Label plot
ax[1].set_xlabel('time (min)')
ax[0].set_ylabel(r'$\left\langle \right.$mRNA$\left. \right\rangle$/cell')
ax[1].set_ylabel(r'$\left\langle \right.$protein$\left. \right\rangle$/cell')

# Align y axis labels
fig.align_ylabels()

# Set legend for both plots
ax[0].legend(loc='upper left', ncol=2, frameon=False,
             bbox_to_anchor=(-0.12, 0, 0, 1.3), fontsize=6.5)

plt.subplots_adjust(hspace=0.05)
plt.savefig(figdir + 'mean_std_cycles.png', bbox_inches='tight',
            transparent=True)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_37_0.png)


We can see from tis plot that the mRNA effectively reaches steady state for each stage of the cell cycle. This is because the degradation rate is fast enough such that the relaxation time is much shorter than the length of the cell cycle. This is not the case for the protein since it never reaches the steady-state expression level. Nevertheless it is very interesting that the cycles reach a "dynamical steady-state" in which the trajectories over cell cycles are reproducible.

# Moment dynamics with cells exponentially distributed along cell cycle

As first discussed by Powell in 1956 populations of cells in a log-phase are exponentially distributed along the cell cycle. This distribution is of the form

$$
P(a) = (\ln 2) \cdot 2^{1 - a},
\tag{4}
$$
where $a \in [0, 1]$ is the stage of the cell cycle, with $a = 0$ being the start of the cycle and $a = 1$ being the division.

Our numerical integration of the moment equations gave us a time evolution of
the moments along the cell cycle. Without loss of generality let's focus on the
first mRNA moment $\ee{m(t)}$ (the same can be applied to all other moments).
In order to calculate the first moment along the entire cell cycle we must
average each time point by the corresponding probability that a cell is found
in such time point. This translates to computing the integral

$$
  \ee{m} = \int_{\text{beginning cell cycle}}^{\text{end cell cycle}}
                       \ee{m(t)} P(t) dt.
\tag{5}
$$

If we map each time point in the cell cycle into a fraction we can use
the distribution and compute instead

$$
  \ee{m} = \int_0^1 \ee{m(a)} P(a) da.
\tag{6}
$$

### Systematically varying the mean protein per mRNA

One thing that we need to test is how sensitive our calculations are to the chosen mean protein copy number per mRNA. For this we will compre the noise (STD / mean) over the entire cell cycel for different mean protein per mRNA values.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define mean protein / mRNA to test
mean_pm = [10, 25, 50, 100, 250, 500, 1000]

# Define number of cell cycles
n_cycles = 6

# Define names for dataframe columns
names = ['mean_pm', 'mean_p', 'second_p']
# initlaize dataframe
df_pm = pd.DataFrame(columns=names)

# Loop through mean protein per mRNA
for i, p_m in enumerate(mean_pm):
    print(p_m)
    # Define initial conditions for integration
    # Single promoter
    gp_init = 1 / (60 * 60)
    rp_init = p_m * gp_init
    # Substitute value of parameters on matrix
    ##  Initial conditions
    A_mat_unreg_s_init = A_mat_unreg_lam(param['kp_on'], param['kp_off'],
                                         param['rm'], param['gm'], 
                                         rp_init, gp_init)
    # Define time on which to perform integration
    t = np.linspace(0, 4000 * 60, 2000)

    # Define initial conditions
    mom_init = np.zeros(len(expo) * 2)
    # Set initial condition for zero moment
    # Since this needs to add up to 1
    mom_init[0] = 1

    # Numerically integrate equations
    mp_sol = sp.integrate.odeint(dmomdt, mom_init, t, 
                                 args=(A_mat_unreg_s_init,))

    mp_init = mp_sol[-1, :]
    
    # Find protein production rate
    rp_opt = sp.optimize.minimize_scalar(rp_residual, 
                                     bounds=(0, 0.1), method='bounded',
                                     args=(mp_init, p_m))

    # Extract parameter
    rp = rp_opt.x
    

    # Define list of parameters
    par_single = [param['kp_on'], param['kp_off'], param['rm'], param['gm'],
              rp, 0]
    par_double = [param['kp_on'], param['kp_off'], 2 * param['rm'],
              param['gm'], rp, 0]

    # Integrate moment equations
    df_p_unreg = dmomdt_cycles(mp_init, t_single, t_double, 
                               A_mat_unreg_lam, 
                               par_single, par_double, expo,
                               n_cycles, Z_mat, n_steps=10000)

    # Extract index for protein first moment
    first_mom_names_p = [x for x in df_p_unreg.columns if 'm0p1' in x]

    # Extract the last cycle information
    df_p_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      first_mom_names_p]
    
    # Extract index for protein second moment
    second_mom_names_p = [x for x in df_p_unreg.columns if 'm0p2' in x]
    
    # Extract the last cycle information
    df_p_unreg_second = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      second_mom_names_p]
    
    # Extract time of last cell cycle
    time = np.sort(df_p_unreg.loc[df_p_unreg.cycle == 
                                  df_p_unreg.cycle.max(),
                                  'time'].unique())

    # Compute the time differences
    time_diff = np.diff(time)

    # Compute the cumulative time difference
    time_cumsum = np.cumsum(time_diff)
    time_cumsum = time_cumsum / time_cumsum[-1]

    # Define array for spacing of cell cycle
    a_array = np.zeros(len(time))
    a_array[1:] = time_cumsum 

    # Compute probability based on this array
    p_a_array = np.log(2) * 2**(1 - a_array)

    # Perform numerical integration
    p_mean = sp.integrate.simps(df_p_unreg_first.sum(axis=1) * p_a_array,
                                      a_array)
    p_second = sp.integrate.simps(df_p_unreg_second.sum(axis=1) * 
                                        p_a_array, a_array)
    
    # Save results on pandas Series
    series = pd.Series([p_m, p_mean, p_second], index=names)
    # Append to dataframe
    df_pm = df_pm.append(series, ignore_index=True)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    10
    25
    50
    100
    250
    500
    1000


Having run the dynamics let's compute the noise


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Compute variance deviation
df_pm['var_p'] = df_pm['second_p'] - df_pm['mean_p']**2

# Compute the standard deviation
df_pm['std_p'] = np.sqrt(df_pm['var_p'])

# Compute the noise
df_pm['noise_p'] = df_pm['std_p'] / df_pm['mean_p']

plt.plot(df_pm['mean_p'], df_pm['noise_p'], '.')
plt.xlabel('mean protein')
plt.ylabel('noise in protein')
plt.ylim([0, 0.25])
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



    (0, 0.25)




![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_45_1.png)


These values seem pretty robust. So at the level of the noise there is not a significant difference with the amount of protein production.

### Systematically varying the mRNA lifetime

Another parameter that we don't directly measure that could have an effect on the protein noise is the mRNA half-life. Let's systematically vary this parameter for the same protein copy number and see if there is an effect.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
1 / param['gm']
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



    180.0




   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define mean protein / mRNA to test
p_m = 500

# Define number of cell cycles
n_cycles = 6

gm_array = [1 / 180, 1 / 360, 1 / 720, 1 / 1440]

# Define names for dataframe columns
names = ['gm', 'mean_p', 'second_p']
# initlaize dataframe
df_gm = pd.DataFrame(columns=names)

# Loop through mean protein per mRNA
for i, g_m in enumerate(gm_array):
    print(g_m)
    # Define initial conditions for integration
    # Single promoter
    gp_init = 1 / (60 * 60)
    rp_init = p_m * gp_init
    # Substitute value of parameters on matrix
    ##  Initial conditions
    A_mat_unreg_s_init = A_mat_unreg_lam(param['kp_on'], param['kp_off'],
                                         param['rm'], g_m, 
                                         rp_init, gp_init)
    # Define time on which to perform integration
    t = np.linspace(0, 4000 * 60, 2000)

    # Define initial conditions
    mom_init = np.zeros(len(expo) * 2)
    # Set initial condition for zero moment
    # Since this needs to add up to 1
    mom_init[0] = 1

    # Numerically integrate equations
    mp_sol = sp.integrate.odeint(dmomdt, mom_init, t, 
                                 args=(A_mat_unreg_s_init,))

    mp_init = mp_sol[-1, :]
    
    # Find protein production rate
    rp_opt = sp.optimize.minimize_scalar(rp_residual, 
                                     bounds=(0, 0.1), method='bounded',
                                     args=(mp_init, p_m))

    # Extract parameter
    rp = rp_opt.x
    

    # Define list of parameters
    par_single = [param['kp_on'], param['kp_off'], param['rm'], g_m,
              rp, 0]
    par_double = [param['kp_on'], param['kp_off'], 2 * param['rm'],
              g_m, rp, 0]

    # Integrate moment equations
    df_p_unreg = dmomdt_cycles(mp_init, t_single, t_double, 
                               A_mat_unreg_lam, 
                               par_single, par_double, expo,
                               n_cycles, Z_mat, n_steps=10000)

    # Extract index for protein first moment
    first_mom_names_p = [x for x in df_p_unreg.columns if 'm0p1' in x]

    # Extract the last cycle information
    df_p_unreg_first = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      first_mom_names_p]
    
    # Extract index for protein second moment
    second_mom_names_p = [x for x in df_p_unreg.columns if 'm0p2' in x]
    
    # Extract the last cycle information
    df_p_unreg_second = df_p_unreg.loc[df_p_unreg.cycle == 
                                      df_p_unreg.cycle.max(),
                                      second_mom_names_p]
    
    # Extract time of last cell cycle
    time = np.sort(df_p_unreg.loc[df_p_unreg.cycle == 
                                  df_p_unreg.cycle.max(),
                                  'time'].unique())

    # Compute the time differences
    time_diff = np.diff(time)

    # Compute the cumulative time difference
    time_cumsum = np.cumsum(time_diff)
    time_cumsum = time_cumsum / time_cumsum[-1]

    # Define array for spacing of cell cycle
    a_array = np.zeros(len(time))
    a_array[1:] = time_cumsum 

    # Compute probability based on this array
    p_a_array = np.log(2) * 2**(1 - a_array)

    # Perform numerical integration
    p_mean = sp.integrate.simps(df_p_unreg_first.sum(axis=1) * p_a_array,
                                      a_array)
    p_second = sp.integrate.simps(df_p_unreg_second.sum(axis=1) * 
                                        p_a_array, a_array)
    
    # Save results on pandas Series
    series = pd.Series([g_m, p_mean, p_second], index=names)
    # Append to dataframe
    df_gm = df_gm.append(series, ignore_index=True)
    
df_gm
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    0.005555555555555556
    0.002777777777777778
    0.001388888888888889
    0.0006944444444444445





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
      <th>gm</th>
      <th>mean_p</th>
      <th>second_p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.005556</td>
      <td>7732.542901</td>
      <td>6.240007e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.002778</td>
      <td>14790.494744</td>
      <td>2.280964e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.001389</td>
      <td>27041.169951</td>
      <td>7.616656e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000694</td>
      <td>45039.874348</td>
      <td>2.111805e+09</td>
    </tr>
  </tbody>
</table>
</div>




   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Compute variance deviation
df_gm['var_p'] = df_gm['second_p'] - df_gm['mean_p']**2

# Compute the standard deviation
df_gm['std_p'] = np.sqrt(df_gm['var_p'])

# Compute the noise
df_gm['noise_p'] = df_gm['std_p'] / df_gm['mean_p']

plt.plot(df_gm['mean_p'], df_gm['noise_p'], '.')
plt.xlabel('mean protein')
plt.ylabel('noise in protein')
plt.ylim([0, 0.25])
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>



    (0, 0.25)




![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_51_1.png)


This has literally no effect on the protein noise.

## Three-state promoter

Let's now include the regulation given by the repressor. For this we change to the three-state promoter that includes two new parameters $\kron$ and $\kroff$. At the mRNA level by assumption our parameter inference assumed that the promoter quickly relaxed from the steady state with one promoter to the steady state with two promoters. This is clearly reflected in the dynamics as we saw in the previous section. Therefore, if we ignore the transients between the single promoter and the two promoters state, the fold-change is of the form. 

$$
\foldchange = 
{ f \cdot \ee{m(R \neq 0)}_1 + (1 - f) \cdot \ee{m(R \neq 0)}_2
\over
f \cdot \ee{m(R = 0)}_1 + (1 - f) \cdot \ee{m(R = 0)}_2},
\tag{7}
$$
where $f \in [0, 1]$ is the fraction of the cell cycle that cells spend with a single copy of the promoter.

Just as our reporter gene changes in copy number and therefore the protein copy number changes along the cell cycle we expect the repressor copy number itself to vary as cells grow and divide. We simplify this picture and assume that the experimentally determined repressor copy number is an effective parameter that remains unchanged along the cell cycle. What that means for our model is that $\kron$ doesn't change along the cell cycle. This is obviously an approximation and only the numerical test of this assumption will tell us how much it affects the theoretical predictions. Under this assumption it can be shown that the fold-change can be simplified to

$$
\foldchange = \left( 1 + {\kron \over \kroff} \left( {\kpon \over \kpon + \kpoff} \right) \right)^{-1}.
\tag{8}
$$
We can then use the fact that the functional form is exactly the same as the thermodynamic fold-change to constraint the value of the $\kron \over \kroff$ ratio.

Working with this let's compute the fold-change using this kinetic model.

### IPTG titration 

To compare the results from the kinetic and the equilibrium model we need to compute the mRNA and protein first moment averaged over the entire cell cycle. We will do this by performing the integral explained in the previous section.

We must now import the matrix $\bb{A}$ for the three-state regulated promoter.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read protein ununregulated matrix 
with open('./pkl_files/three_state_protein_dynamics_matrix.pkl', 'rb') as file:
    A_mat_reg_lam = cloudpickle.load(file)
    expo_reg = cloudpickle.load(file)
```

Now that we know that the rates are able to reproduce the equilibrium picture of the LacI titration (up to a systematic deviation) let's complete the analysis of the equivalence between both frameworks by including the effect of the inducer.  For this analysis we will keep the operator fix and vary both the repressor copy number and the IPTG concentration.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# repressor-DNA binding energy
op = "O2"
eRA = -13.9  # kBT

# Define repressor copy number list
rep_array = [22, 260, 1740]  # repressors per cell

# Define IPTG concentrations
iptg_array = [0, 0.1, 25, 50, 500, 1000]  # µM

# Initialize data frame to save fold-changes
names = [
    "operator",
    "energy",
    "repressors",
    "iptg_uM",
    "mean_m_reg",
    "mean_m_unreg",
    "fold_change_m",
    "mean_p_reg",
    "mean_p_unreg",
    "fold_change_p",
]
df_fc_iptg = pd.DataFrame(columns=names)

# Loop through operators
for j, iptg in enumerate(iptg_array):
    print(iptg)
    # Loop through repressor copy numbers
    for i, rep in enumerate(rep_array):
        # Define parameters
        eRA = param[f"epR_{op}"]
        kp_on = param["kp_on"]
        kp_off = param["kp_off"]
        kr_off = param["kr_off_O2"]
        ko = param["k0"]
        rm = param["rm"]
        gm = param["gm"]
        rp = param["rp"]
        ka = param["Ka"]
        ki = param["Ki"]
        epAI = param["epAI"]

        # Calculate the repressor on rate including the MWC model
        kr_on = ko * rep * ccutils.model.p_act(iptg, ka, ki, epAI)

        # Generate matrices for dynamics
        # Single promoter
        par_reg_s = [kr_on, kr_off, kp_on, kp_off, rm, gm, rp, 0]
        # Two promoters
        par_reg_d = [kr_on, kr_off, kp_on, kp_off, 2 * rm, gm, rp, 0]

        # Initial conditions
        A_reg_s_init = A_mat_reg_lam(
            kr_on, kr_off, kp_on, kp_off, rm, gm, rp_init, gp_init
        )

        # Define initial conditions
        mom_init = np.zeros(len(expo_reg) * 3)
        # Set initial condition for zero moment
        # Since this needs to add up to 1
        mom_init[0] = 1

        # Define time on which to perform integration
        t = np.linspace(0, 4000 * 60, 10000)
        # Numerically integrate equations
        m_init = sp.integrate.odeint(dmomdt, mom_init, t, args=(A_reg_s_init,))
        # Keep last time point as initial condition
        m_init = m_init[-1, :]

        # Integrate moment equations
        df = dmomdt_cycles(
            m_init,
            t_single,
            t_double,
            A_mat_reg_lam,
            par_reg_s,
            par_reg_d,
            expo_reg,
            n_cycles,
            Z_mat,
            states=["A", "I", "R"],
            n_steps=3000,
        )

        # Keep only last cycle
        df = df[df["cycle"] == df["cycle"].max()]

        # Extract index for first moment
        first_mom_names_m = [x for x in df.columns if "m1p0" in x]
        first_mom_names_p = [x for x in df.columns if "m0p1" in x]

        # Extract the last cycle information of the first moments
        df_m_reg_first = df.loc[:, first_mom_names_m]
        df_p_reg_first = df.loc[:, first_mom_names_p]

        # Extract time of last cell cycle
        time = np.sort(df["time"].unique())

        # Compute the time differences
        time_diff = np.diff(time)
        # Compute the cumulative time difference
        time_cumsum = np.cumsum(time_diff)
        time_cumsum = time_cumsum / time_cumsum[-1]

        # Define array for spacing of cell cycle
        a_array = np.zeros(len(time))
        a_array[1:] = time_cumsum
        # Compute probability based on this array
        p_a_array = np.log(2) * 2 ** (1 - a_array)

        # Perform numerical integration
        m_mean_reg = sp.integrate.simps(
            df_m_reg_first.sum(axis=1) * p_a_array, a_array
        )
        p_mean_reg = sp.integrate.simps(
            df_p_reg_first.sum(axis=1) * p_a_array, a_array
        )

        # Compute the fold-change
        fold_change_m = m_mean_reg / m_mean_unreg
        fold_change_p = p_mean_reg / p_mean_unreg

        # Save results into series in order to append it to data frame
        series = pd.Series(
            [
                op,
                eRA,
                rep,
                iptg,
                m_mean,
                m_mean_unreg,
                fold_change_m,
                p_mean,
                p_mean_unreg,
                fold_change_p,
            ],
            index=names,
        )

        df_fc_iptg = df_fc_iptg.append(series, ignore_index=True)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    0
    0.1
    25
    50
    500
    1000


Let's plot the inducer titration to compare the predictions done by the equilibrium picture and the kinetic model with this new variation.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define IPTG range to compute thermodynamic fold-change
iptg = np.logspace(-1, 3, 50)
iptg_lin = [0, 0.1]

# Group data frame by repressor copy number
df_group = df_fc_iptg.groupby('repressors')

# Define colors
colors = sns.color_palette('colorblind', n_colors=len(df_group))

# Loop through each of the repressor copy numbers
for i, (rep, data) in enumerate(df_group):
    Nns = param['Nns']
    # Compute thermodynamic fold-change
    fc_thermo = (1 + rep / Nns * ccutils.model.p_act(iptg, ka, ki, epAI) *
                 np.exp(- data.energy.unique()[0]))**-1
    fc_thermo_lin = (1 + rep / Nns * ccutils.model.p_act(iptg_lin, 
                                                         ka, ki, epAI) *
                 np.exp(- data.energy.unique()[0]))**-1
    
    # Plot thermodynamic fold-change prediction
    plt.plot(iptg, fc_thermo, label=str(rep), color=colors[i])
    plt.plot(iptg_lin, fc_thermo_lin, label='', color=colors[i],
             linestyle='--')
    
    # Plot the kinetic fold-change prediciton
    # Protein
    plt.plot(data.iptg_uM.values, data.fold_change_p.values, lw=0, marker='o', 
             color=colors[i], label='')
    # mRNA
    plt.plot(data.iptg_uM.values, data.fold_change_m.values, lw=0, marker='v', 
             markeredgecolor=colors[i], markeredgewidth=1,
             markerfacecolor='w', label='')

# Generate labels for mRNA and protein
plt.plot([], [], lw=0, marker='v', 
         markeredgecolor='k', markeredgewidth=1,
         markerfacecolor='w', label='mRNA')
plt.plot([], [], lw=0, marker='o', 
         color='k', label='protein')
    
# Change scale to log
plt.xscale('symlog', linthreshx=1E-1, linscalex=0.5)

# Label axis
plt.xlabel(r'IPTG ($\mu$M)')
plt.ylabel('fold-change')

# Set legend
legend = plt.legend(title=r'$\beta\Delta\epsilon_r =  -13.5$' '\n rep. / cell',
                    fontsize=5)
plt.setp(legend.get_title(),fontsize=6)

# Save figure
plt.tight_layout()
plt.savefig(figdir + 'IPTG_titration.pdf', bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_62_0.png)


This looks pretty good. It seems that at the level of mean gene expression the kinetic model can reproduce the predictions made by the thermodynamic model.

## Systematic moment computation

Now that we confirmed that these parameters can reproduce the equilibrium picture let's systematically obtain average moments for varying repressor copy numbers, operators and inducer concentrations that later on we will use to compute the maximum entropy approximation of the distribution.

We will generate and export a tidy data frame containing all moments.

### Varying IPTG concenrations, experimental repressor copy number.

On a separate script `src/theory/scripts/mdcd_iptg_range.py` we compute the moments for a fine grid of IPTG concentrations. Here we will just load the resulting tidy data frame.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
df_mom_iptg = pd.read_csv(datadir + 'MaxEnt_multi_prom_IPTG_range.csv')
df_mom_iptg.head()
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
      <th>m0p0</th>
      <th>m1p0</th>
      <th>m2p0</th>
      <th>m3p0</th>
      <th>m4p0</th>
      <th>m5p0</th>
      <th>...</th>
      <th>m3p2</th>
      <th>m2p3</th>
      <th>m1p4</th>
      <th>m0p5</th>
      <th>m5p1</th>
      <th>m4p2</th>
      <th>m3p3</th>
      <th>m2p4</th>
      <th>m1p5</th>
      <th>m0p6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0</td>
      <td>0.100000</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0</td>
      <td>0.125284</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0</td>
      <td>0.156961</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0</td>
      <td>0.196646</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



### Varying number of repressors, 12 IPTG concentrations.

As in the previous case a separate script `src/theory/scripts/mdcd_repressor_range.py` we compute the moments for a fine grid of IPTG concentrations. Here we will just load the resulting tidy data frame.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
df_mom_rep = pd.read_csv(datadir + 'MaxEnt_multi_prom_constraints.csv')
df_mom_rep.head()
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
      <th>m0p0</th>
      <th>m1p0</th>
      <th>m2p0</th>
      <th>m3p0</th>
      <th>m4p0</th>
      <th>m5p0</th>
      <th>...</th>
      <th>m3p2</th>
      <th>m2p3</th>
      <th>m1p4</th>
      <th>m0p5</th>
      <th>m5p1</th>
      <th>m4p2</th>
      <th>m3p3</th>
      <th>m2p4</th>
      <th>m1p5</th>
      <th>m0p6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>0.1</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O1</td>
      <td>-15.3</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>1.000001</td>
      <td>15.46507</td>
      <td>322.675729</td>
      <td>8592.936386</td>
      <td>278504.362303</td>
      <td>1.054893e+07</td>
      <td>...</td>
      <td>7.435490e+11</td>
      <td>2.400384e+14</td>
      <td>9.106554e+16</td>
      <td>4.227172e+19</td>
      <td>1.023094e+11</td>
      <td>2.564953e+13</td>
      <td>7.223464e+15</td>
      <td>2.324187e+18</td>
      <td>8.751674e+20</td>
      <td>4.007275e+23</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



## Comparing constraints with single-promoter model.

An interesting question is how we expect the moments to change with respect to a kinetic model in which the variability in gene copy number along the cell cycle is ignored.

# Noise

Let's now look at the noise defined as

$$
\text{noise} \equiv {\text{STD}(X) \over \ee{X}},
\tag{9}
$$
where $\text{STD}(x)$ is the standard deviation of the random variable $X$. The reason for choosing this metric over the commonly used Fano factor is that when quantified from experimental data this is a dimensionless quantity that can be directly inferred from arbitrary units of fluorescence as long as there is a linear relationship between these arbitrary units and the absolute molecule count.

The expectation here is that since having multiple promoters increases the variability over the cell cycle, the multi-promoter model should have a higher noise.

Let's first compute this quantity along with the fold-change for both the mRNA and protein level


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Read constraints for the single promoter model
df_mom_single = pd.read_csv(datadir + 'single_prom_moments.csv')

# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m1p0
)
mean_p_delta = np.mean(
    df_mom_iptg[df_mom_iptg.repressor==0].m0p1
)

# Compute the noise for the multi-promoter data
df_mom_iptg = df_mom_iptg.assign(
    m_noise=np.sqrt(df_mom_iptg.m2p0 - df_mom_iptg.m1p0**2) / 
            df_mom_iptg.m1p0,
    p_noise=np.sqrt(df_mom_iptg.m0p2 - df_mom_iptg.m0p1**2) / 
            df_mom_iptg.m0p1,
    m_fold_change=df_mom_iptg.m1p0 / mean_m_delta,
    p_fold_change=df_mom_iptg.m0p1 / mean_p_delta
)
```

Now let's plot the noise for the regulated case. We will show the difference between the single and the multiple promoter model for different operators (repressor-DNA binding energy) and varying repressor copy numbers.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Define repressor copy numbers to include
rep = [22, 260, 1740]

# Group moments by operator and repressor
df_group = df_mom_iptg[df_mom_iptg['repressor'].isin(rep)].\
           sort_values('inducer_uM').\
           groupby(['operator', 'repressor'])

df_group_single = df_mom_single[df_mom_single['repressor'].\
                  isin(rep)].sort_values('inducer_uM').\
                  groupby(['operator', 'repressor'])

# Generate index for each opeartor
operators = ['O1', 'O2', 'O3']
op_idx = dict(zip(operators, np.arange(3)))

# Define energies to go along operators
energies = [-15.3, -13.9, -9.7]

# Generate list of colors
col_list = ['Blues_r', 'Oranges_r', 'Greens_r']
# Loop through operators generating dictionary of colors for each
col_dict = {}
for i, op in enumerate(operators):
    col_dict[op] = dict(zip(rep, sns.color_palette(col_list[i],
                                 n_colors=len(rep) + 3)[0:3]))

# Define threshold to separate linear from logarithmic scale
thresh = 1E-1

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]]].plot(data[data.inducer_uM >= thresh].inducer_uM, 
                              data[data.inducer_uM >= thresh].p_noise, 
                              color=col_dict[group[0]][group[1]],
                              label=int(group[1]))
    # linear scale
    ax[op_idx[group[0]]].plot(data[data.inducer_uM <= thresh].inducer_uM, 
                              data[data.inducer_uM <= thresh].p_noise, 
                              color=col_dict[group[0]][group[1]],
                              label='', linestyle=':')

# Loop through groups on single-promoter
for i, (group, data) in enumerate(df_group_single):
    # Log scale
    ax[op_idx[group[0]]].plot(data[data.inducer_uM >= thresh].inducer_uM, 
                              data[data.inducer_uM >= thresh].p_noise, 
                              linestyle='--',
                              color=col_dict[group[0]][group[1]],
                              label='', alpha=1)
    # Linear scale
    ax[op_idx[group[0]]].plot(data[data.inducer_uM <= thresh].inducer_uM, 
                              data[data.inducer_uM <= thresh].p_noise, 
                              linestyle=':',
                              color=col_dict[group[0]][group[1]],
                              label='', alpha=1)

# Define location for secondary legend
leg2_loc = ['lower left'] * 2 + ['upper left']
for i, a in enumerate(ax):
    # Generate legend for single vs double promoter
    single, = ax[i].plot([], [], color='k', linestyle='--', label='',
               alpha=1)
    multi, = ax[i].plot([], [], color='k', label='')
    # systematically change axis for all subplots
    ax[i].set_xscale('symlog', linthreshx=1E-1, linscalex=1)
    ax[i].set_yscale('log')
    ax[i].set_ylim(top=10)
    # Set legend
    leg1 = ax[i].legend(title='rep./cell', fontsize=7, loc='upper right')
    # Set legend font size
    plt.setp(leg1.get_title(), fontsize=7)
    # leg1 will be removed from figure
    leg2 = ax[i].legend([multi, single], ['multiple', 'single'],
                        loc=leg2_loc[i],
                        fontsize=6, title='# promoters')
    # Set legend font size
    plt.setp(leg2.get_title(), fontsize=6)
    # Manually add the first legend back
    ax[i].add_artist(leg1)
 
    # Set title
    label = r'$\Delta\epsilon_r$ = {:.1f} $k_BT$'.\
                    format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor='#ffedce'))
    # Label axis
    ax[i].set_xlabel(r'IPTG (µM)')
ax[0].set_ylabel(r'noise')

# Change spacing between plots
plt.subplots_adjust(wspace=0.05)

plt.savefig(figdir + 'noise_comparison.pdf', bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_79_0.png)


We can see that there is a striking difference between both models, especially at high inducer concentrations. This shows that there is a significant amount of cell-to-cell variability that comes from the gene copy number variability and the non-steady-state dynamics of the moments according to the model.

## Comparison with experimental data

In order to assess if our dynamical theory for the moments works we will compare experimentally determined moments to our theoretical predictions.

Consider that the noise is defined as

\begin{equation}
\text{noise} \equiv \frac{\sqrt{\left\langle p^2 \right\rangle - \left\langle p \right\rangle^2}}{\left\langle p \right\rangle}.
\tag{10}
\end{equation}
Assume that the intensity level of a cell $I$ is linearly proportional to the absolute protein count, i.e.

$$
I = \alpha p,
\tag{11}
$$
where $\alpha$ is the proportionality constant between arbitrary units (a.u.) and protein count. Substituting this definition on the noise gives

\begin{equation}
\text{noise} = {\sqrt{\left\langle (\alpha I)^2 \right\rangle - 
                   \left\langle \alpha I \right\rangle^2} \over 
                   \left\langle \alpha I \right\rangle}.
\tag{12}
\end{equation}
Since $\alpha$ is a constant it can be taken out of the average operator $\ee{\cdot}$, obtaining

\begin{equation}
\text{noise} = {\sqrt{\alpha^2 \left(\left\langle I^2 \right\rangle - 
              \left\langle I \right\rangle^2 \right)} \over 
              \alpha \left\langle  I \right\rangle}
     = {\sqrt{\left(\left\langle I^2 \right\rangle - 
              \left\langle I \right\rangle^2 \right)} \over 
              \left\langle  I \right\rangle}
\tag{13}
\end{equation}

The proportionality between intensity and protein count has no intercept. This ignores the autofluorescence that cells without
reporter would generate. Therefore in practice to compute the noise from experimental intensity measurements we compute

\begin{equation}
\text{noise} = \frac{\sqrt{\left\langle (I  - \langle I_{\text{auto}}\rangle)^2 \right\rangle - \left\langle (I  - \langle I_{\text{auto}}\rangle) \right\rangle^2}}{\left\langle (I  - \langle I_{\text{auto}}\rangle) \right\rangle},
\tag{14}
\end{equation}
where $I$ is the intensity of the objective strain and $\langle I_{\text{auto}}\rangle$ is the mean autofluorescence intensity.

Having shown that this quantity is dimensionless we can therefore compare the experimentally determined noise with our theoretical predictions.

Note: For this noise we have already computed a bootstrap estimate of the error (See `src/image_analsysis/scripts/`). We will import here the noise estimates for all strains.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
df_noise = pd.read_csv('../../data/csv_microscopy/' + 
                       'microscopy_noise_bootstrap.csv')

df_noise.head()
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
      <th>Unnamed: 0</th>
      <th>date</th>
      <th>IPTG_uM</th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>percentile</th>
      <th>fold_change</th>
      <th>fold_change_lower</th>
      <th>fold_change_upper</th>
      <th>noise</th>
      <th>noise_lower</th>
      <th>noise_upper</th>
      <th>skewness</th>
      <th>skewness_lower</th>
      <th>skewness_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.371102</td>
      <td>0.371409</td>
      <td>0.825615</td>
      <td>0.802211</td>
      <td>0.805069</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.370936</td>
      <td>0.372769</td>
      <td>0.825615</td>
      <td>0.797166</td>
      <td>0.816022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.368780</td>
      <td>0.372603</td>
      <td>0.825615</td>
      <td>0.793003</td>
      <td>0.832204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.363071</td>
      <td>0.373106</td>
      <td>0.825615</td>
      <td>0.793003</td>
      <td>0.894289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.358524</td>
      <td>0.379618</td>
      <td>0.825615</td>
      <td>0.712660</td>
      <td>0.931086</td>
    </tr>
  </tbody>
</table>
</div>



### Unregulated promoter

Let's first take the intensity measurements of the $\Delta lacI$ strains and compute the noise.

Now let's plot the noise for each of the operators along with the theoretical prediction for the multi-promoter model. In principle there shouldn't be any difference between operators since these are all unregulated promoters. But it it known that basepairs downstream the RNAP binding site can affect transcriptional output as well. We ignore this in the model, but to make sure here we will plot each $\Delta lacI$ strain separatade by operators.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract theoretical noise for the ∆lacI strain
noise_delta_thry = df_mom_iptg[df_mom_iptg.repressor == 0].p_noise.mean()
noise_delta_thry_single = df_mom_single[
    df_mom_single.repressor == 0
].p_noise.mean()

# Extract data with 95% percentile
df_delta = df_noise[(df_noise.repressor == 0) & (df_noise.percentile == 0.95)]

# Define colors for operators
col_list = ["Blues_r", "Reds_r", "Greens_r"]
colors = [sns.color_palette(x, n_colors=1) for x in col_list]

# Plot theoretical prediction

# Generate stripplot for experimentally determined
# noise of the ∆lacI strain
fig, ax = plt.subplots(1, 1)
ccutils.viz.jitterplot_errorbar(ax, df_delta, jitter=0.1)

# Plot theoretical prediction as a horizontal black line
ax.axhline(
    noise_delta_thry_single,
    color="gray",
    linestyle=":",
    label="single-promoter",
)
ax.axhline(noise_delta_thry, color="k", linestyle="--", label="multi-promoter")

# Include legend
ax.legend(title="model", loc="upper center")

# Set axis limits
ax.set_ylim([0, 1])

# Label axis
ax.set_ylabel(r"noise")

# Save figure
plt.tight_layout()
plt.savefig(figdir + "noise_delta_microscopy.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_88_0.png)


The prediction are not entirely correct for either model. But the multi-promoter model is a little closer to the data.

### Regulated promoter

Let's extend the analysis to the regulated promoter.

Let's now plot the noise as a function of the IPTG concentration for all strains measured experimentally. Here we will show with a solid line the predictions made by the model that accoutns for gene copy number variability during the cell cycle, and with a dotted line the predictions for the single promoter model.

But first let's generate the groups that we will need, as well as the color palettes that we will use.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract regulated promoter information
df_noise_reg = df_noise[df_noise.repressor > 0]
# Define repressor copy numbers to include
rep = df_noise_reg["repressor"].unique()

# Group moments by operator and repressor
df_group_exp = (
    df_noise_reg[df_noise_reg.noise > 0]
    .sort_values("IPTG_uM")
    .groupby(["operator", "repressor"])
)

df_group = (
    df_mom_iptg[df_mom_iptg["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

df_group_single = (
    df_mom_single[df_mom_single["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

# Generate index for each opeartor
operators = ["O1", "O2", "O3"]
op_idx = dict(zip(operators, np.arange(3)))

# Generate list of colors
col_list = ["Blues_r", "Oranges_r", "Greens_r"]
# Loop through operators generating dictionary of colors for each
col_dict = {}
for i, op in enumerate(operators):
    col_dict[op] = dict(
        zip(rep, sns.color_palette(col_list[i], n_colors=len(rep) + 1)[0:3])
    )

# Define threshold to separate log scale from linear scale
thresh = 1e-1
```

Now let's plot the noise. To be fair we should include all noise measurements in the plot, but without drawing too much attention for the ones that are obviously problematic. To solve that issue we will add a secondary axis to include the points with too large deviations.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize figure
fig, ax = plt.subplots(
    2,
    3,
    figsize=(7, 2.5),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1, 5], "wspace": 0.05, "hspace": 0},
)
ax = ax.ravel()
# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # Linear scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Set threshold for data
dthresh = 10
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot data points on lower plot
    ax[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot same data points with different plotting style on the upper row
    ax[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        linestyle="--",
        color="w",
        label="",
        lw=0,
        marker="o",
        markersize=3,
        markeredgecolor=col_dict[group[0]][group[1]],
    )

# Set scales of reference plots and the other ones will follow
ax[0].set_xscale("symlog", linthreshx=thresh, linscalex=1)
ax[0].set_yscale("log")
ax[3].set_yscale("log")

# Set limits of reference plots and the rest will folow
ax[3].set_ylim(top=6)
ax[0].set_ylim([6, 5e2])

# Set ticks for the upper plot
ax[0].set_yticks([1e1, 1e2])

# Define location for secondary legend
leg2_loc = ["lower left"] * 2 + ["upper left"]

for i in range(3):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i + 3].set_xlabel(r"IPTG ($\mu$M)")
    # Set legend
    leg = ax[i + 3].legend(title="rep./cell", fontsize=8)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)
ax[3].set_ylabel(r"noise")

# Save figure
plt.savefig(figdir + "noise_comparison_exp_scale.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_95_0.png)


Let's repeat the same plot, but this time showing the noise in linear scale for a range < 10.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize figure
fig, ax = plt.subplots(
    2,
    3,
    figsize=(7, 2.5),
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [1, 5], "wspace": 0.05, "hspace": 0},
)
ax = ax.ravel()
# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # Linear scale
    ax[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Define threshold for data
dthresh = 7
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot data points on lower plot
    ax[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot same data points with different plotting style on the upper row
    ax[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        linestyle="--",
        color="w",
        label="",
        lw=0,
        marker="o",
        markersize=3,
        markeredgecolor=col_dict[group[0]][group[1]],
    )

# Set scales of reference plots and the other ones will follow
ax[0].set_xscale("symlog", linthreshx=thresh, linscalex=1)
ax[0].set_yscale("log")

# Set limits of reference plots and the rest will folow
ax[3].set_ylim([-0.5, dthresh])
ax[0].set_ylim([dthresh, 5e2])

# Set ticks
ax[3].set_yticks([0, 2, 4, 6])
ax[0].set_yticks([1e1, 1e2])

# Define location for secondary legend
leg2_loc = ["lower left"] * 2 + ["upper left"]

for i in range(3):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i + 3].set_xlabel(r"IPTG ($\mu$M)")
    # Set legend
    leg = ax[i + 3].legend(title="rep./cell", fontsize=8)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)
ax[3].set_ylabel(r"noise")

# Save figure
plt.savefig(figdir + "noise_comparison_lin_scale.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_97_0.png)


### fold-change & noise side to side

Let's now look at the fold-change and the noise simultaneously to show that the theory can capture both the first and the second moment.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
fig = plt.figure(figsize=(5, 3))
# Define outer grispec to keep at top the fold-change and at the bottom
# the noise
gs_out = mpl.gridspec.GridSpec(
    2, 1, height_ratios=[1, 1 + 1 / 5], hspace=0.1, wspace=0.05
)

# make nested gridspecs
gs_fc = mpl.gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_out[0], wspace=0.05
)
gs_noise = mpl.gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=gs_out[1],
    wspace=0.05,
    hspace=0.01,
    height_ratios=[1, 5],
)

# Add axis to plots
# fold-change
ax_fc = [plt.subplot(gs) for gs in gs_fc]
# noise
ax_noise = [plt.subplot(gs) for gs in gs_noise]

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Plot fold-change
    # Linear
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

    # Plot noise
    # Linear
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

# Define data threshold
dthresh = 7
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot fold_change
    ax_fc[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.fold_change,
        yerr=[
            data.fold_change - data.fold_change_lower,
            data.fold_change_upper - data.fold_change,
        ],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        color="w",
        markeredgecolor=col_dict[group[0]][group[1]],
        label="",
        lw=0,
        marker="o",
        markersize=2,
    )

##  Set shared axis

# fold-change
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_fc[i]
    # join axis with first plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_fc[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Remove x ticks from left plot
plt.setp(ax_fc[0].get_xticklabels(), visible=False)
# Set axis to be shared with left lower plot
ax_fc[0].get_shared_x_axes().join(ax_fc[0], ax_noise[3])

# noise upper
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_noise[i]
    # join x axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    # join y axis with upper left plot
    ax.get_shared_y_axes().join(ax, ax_noise[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Set upper left plot x axis to be shared with lower left plot
ax.get_shared_x_axes().join(ax_noise[0], ax_noise[3])
# Remove x ticks from left plot
plt.setp(ax_noise[0].get_xticklabels(), visible=False)

# noise lower
# Loop through axis
for i in range(4, 6):
    # Select axis
    ax = ax_noise[i]
    # join axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_noise[3])
    # Remove y ticks labels
    plt.setp(ax.get_yticklabels(), visible=False)

# Set scales of reference plots and the other ones will follow
ax_noise[3].set_xscale("symlog", linthreshx=thresh)  # , linscalex=0.5)
ax_noise[0].set_yscale("log")

# Set limits
for i in range(3):
    ax_fc[i].set_ylim([-0.05, 1.4])

ax_noise[0].set_ylim([dthresh, 5e2])
ax_noise[3].set_ylim([-0.25, dthresh])

# Label axis
for i, ax in enumerate(ax_fc):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax.set_title(label, bbox=dict(facecolor="#ffedce"))
    # Set legend
    leg = ax.legend(title="rep./cell", fontsize=5)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=5)
    leg2 = ax_noise[i + 3].legend(
        title="rep./cell", fontsize=5, loc="upper right"
    )
    plt.setp(leg2.get_title(), fontsize=5)

    ax_noise[i + 3].set_xlabel(r"IPTG ($\mu$M)")

# Set ticks for the upper noise plot
ax_noise[0].set_yticks([1e1, 1e2])
ax_noise[1].set_yticks([1e1, 1e2])
ax_noise[2].set_yticks([1e1, 1e2])

# Add y axis labels
ax_fc[0].set_ylabel(r"fold-change")
ax_noise[3].set_ylabel(r"noise")

# Align y axis labels
fig.align_ylabels()

plt.savefig(figdir + "moment_comparison_lin_scale.pdf", bbox_inches="tight")
plt.savefig(figdir + "moment_comparison_lin_scale.svg", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_100_0.png)


there is a clear systematic deviation between the theoretical predictions and the experimental determination of the noise. Our model underestimates the level of cell-to-cell variability for all cases. This is something we need to address.

---
# Exploration of systematic deviation in the noise

Let's take a closer look at how the theoretical prediction for the noise deviates from the experimental result. For this we will plot theory vs experiment directly. Let's first compute the noise for the `df_mom_rep` data frame.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Find the mean unregulated levels to compute the fold-change
mean_m_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m1p0)
mean_p_delta = np.mean(df_mom_rep[df_mom_rep.repressor == 0].m0p1)

# Compute the noise for the multi-promoter data
df_mom_rep = df_mom_rep.assign(
    m_noise=(
        np.sqrt(df_mom_rep.m2p0 - df_mom_rep.m1p0 ** 2) / df_mom_rep.m1p0
    ),
    p_noise=(
        np.sqrt(df_mom_rep.m0p2 - df_mom_rep.m0p1 ** 2) / df_mom_rep.m0p1
    ),
    m_fold_change=df_mom_rep.m1p0 / mean_m_delta,
    p_fold_change=df_mom_rep.m0p1 / mean_p_delta,
)
```

Now we'll loop through each of the experimental measurements and assign the corresponding theoretical noise


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize list to save theoretical noise
thry_noise = list()
# Iterate through rows
for idx, row in df_noise.iterrows():
    # Extract information
    rep = float(row.repressor)
    op = row.operator
    if np.isnan(row.IPTG_uM):
        iptg = 0
    else:
        iptg = row.IPTG_uM
    
    # Extract equivalent theoretical prediction
    thry = df_mom_rep[(df_mom_rep.repressor == rep) &
                       (df_mom_rep.operator == op) &
                       (df_mom_rep.inducer_uM == iptg)].p_noise
    # Append to list
    thry_noise.append(thry.iloc[0])
    
df_noise = df_noise.assign(noise_theory = thry_noise)

df_noise.head()
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
      <th>Unnamed: 0</th>
      <th>date</th>
      <th>IPTG_uM</th>
      <th>operator</th>
      <th>binding_energy</th>
      <th>repressor</th>
      <th>percentile</th>
      <th>fold_change</th>
      <th>fold_change_lower</th>
      <th>fold_change_upper</th>
      <th>noise</th>
      <th>noise_lower</th>
      <th>noise_upper</th>
      <th>skewness</th>
      <th>skewness_lower</th>
      <th>skewness_upper</th>
      <th>noise_theory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.371102</td>
      <td>0.371409</td>
      <td>0.825615</td>
      <td>0.802211</td>
      <td>0.805069</td>
      <td>0.208828</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.370936</td>
      <td>0.372769</td>
      <td>0.825615</td>
      <td>0.797166</td>
      <td>0.816022</td>
      <td>0.208828</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.368780</td>
      <td>0.372603</td>
      <td>0.825615</td>
      <td>0.793003</td>
      <td>0.832204</td>
      <td>0.208828</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.363071</td>
      <td>0.373106</td>
      <td>0.825615</td>
      <td>0.793003</td>
      <td>0.894289</td>
      <td>0.208828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>20161203</td>
      <td>NaN</td>
      <td>O2</td>
      <td>-13.9</td>
      <td>0</td>
      <td>0.50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.370025</td>
      <td>0.358524</td>
      <td>0.379618</td>
      <td>0.825615</td>
      <td>0.712660</td>
      <td>0.931086</td>
      <td>0.208828</td>
    </tr>
  </tbody>
</table>
</div>



Let's now plot the theoretical vs experimental noise.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

# Linear scale

# Plot reference line
ax[0].plot([1e-2, 1e2], [1e-2, 1e2], "--", color="gray")

# Plot error bars
ax[0].errorbar(
    x=df_noise.noise_theory,
    y=df_noise.noise,
    yerr=[
        df_noise.noise - df_noise.noise_lower,
        df_noise.noise_upper - df_noise.noise,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

# Plot data with color depending on log fold-change
ax[0].scatter(
    df_noise.noise_theory,
    df_noise.noise,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[0].set_xlabel("theoretical noise")
ax[0].set_ylabel("experimental noise")
ax[0].set_title("linear scale")

ax[0].set_xlim(0, 4)
ax[0].set_ylim(0, 4)
ax[0].set_xticks([0, 1, 2, 3, 4])
ax[0].set_yticks([0, 1, 2, 3, 4])

# Log scale

# Plot reference line
line = [1e-1, 1e2]
ax[1].loglog(line, line, "--", color="gray")
# Plot data with color depending on log fold-change

ax[1].errorbar(
    x=df_noise.noise_theory,
    y=df_noise.noise,
    yerr=[
        df_noise.noise - df_noise.noise_lower,
        df_noise.noise_upper - df_noise.noise,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

plot = ax[1].scatter(
    df_noise.noise_theory,
    df_noise.noise,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[1].set_xlabel("theoretical noise")
ax[1].set_ylabel("experimental noise")
ax[1].set_title("log scale")
ax[1].set_xlim([0.1, 10])

# show color scale
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot, cax=cbar_ax, ticks=[0, -1, -2, -3])

cbar.ax.set_ylabel("fold-change")
cbar.ax.set_yticklabels(["1", "0.1", "0.01", "0.001"])
cbar.ax.tick_params(width=0)

plt.subplots_adjust(wspace=0.3)
plt.savefig(figdir + "noise_theory_vs_exp.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_108_0.png)


We can see that in general the noise predictions seem to be systematically off by roughly a factor of ≈ 2. There is a general trend that the predictions deviate the most when the fold-change is really small. This is expected since it means that the fluorescence of such cells was really close to the background, therefore the mean fluorescence is ≈ 0. Any small change in this mean is enormously amplified when computing the noise whose denominator is this number close to zero.

## Multiplicative factor for experimental noise

Let's assume that the experimental noise has a multiplicative nature, meaning that the systematic deviations of the theory from the experiment can be explained by multiplying the predictions by a constant factor. By eye if we had to guess this factor would be ≈ 2. But let's do this systematically by performing a linear regression with a fix intercept. Since we have large deviations for some of the values, but those correlate with measurements with extremely small fold-changes, we'll perform a weighted linear regression where each weight is related to the fold-change of each measurement.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract fold-change
fc = df_noise.fold_change.values
# Set values for ∆lacI to be fold-change 1
fc[np.isnan(fc)] = 1
# Normalize weights
weights = fc / fc.sum()

# Declare linear regression model
wls_model = sm.WLS(df_noise.noise.values,
                   df_noise.noise_theory.values,
                   weights=weights)
# Fit parameter
results = wls_model.fit()
print(f'Multiplicative factor: {results.params[0]}')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    Multiplicative factor: 2.038810005532838


Exactly what our eye was telling us, if the systematic deviations were explained by a multiplicative factor, this would be close to a factor of 2.

Let's repeat the previous plot with this multiplicative factor.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

# Linear scale

# Plot reference line
ax[0].plot([1E-2, 1E2], [1E-2, 1E2], '--', color='gray')

# Plot error bars
ax[0].errorbar(x=df_noise.noise_theory * results.params[0],
               y=df_noise.noise,
               yerr=[df_noise.noise - df_noise.noise_lower,
                     df_noise.noise_upper - df_noise.noise],
               color='gray',
               alpha=0.5,
               mew=0,
               zorder=0,
               fmt='.')

# Plot data with color depending on log fold-change
ax[0].scatter(df_noise.noise_theory * results.params[0], df_noise.noise, 
              c=np.log10(df_noise.fold_change), cmap='viridis',
              s=10)

ax[0].set_xlabel('theoretical noise')
ax[0].set_ylabel('experimental noise')
ax[0].set_title('linear scale')

ax[0].set_xlim(0, 4)
ax[0].set_ylim(0, 6);
# ax[0].set_xticks([0, 1, 2, 3, 4, 6])
# ax[0].set_yticks([0, 1, 2, 3, 4, 6])

# Log scale

# Plot reference line
line = [1E-1, 1E2]
ax[1].loglog(line, line, '--', color='gray')
# Plot data with color depending on log fold-change

ax[1].errorbar(x=df_noise.noise_theory * results.params[0],
               y=df_noise.noise,
               yerr=[df_noise.noise - df_noise.noise_lower,
                     df_noise.noise_upper - df_noise.noise],
               color='gray',
               alpha=0.5,
               mew=0,
               zorder=0,
               fmt='.')

plot = ax[1].scatter(df_noise.noise_theory * results.params[0],
                     df_noise.noise, 
                     c=np.log10(df_noise.fold_change), cmap='viridis',
                     s=10)

ax[1].set_xlabel('theoretical noise')
ax[1].set_ylabel('experimental noise')
ax[1].set_title('log scale')
ax[1].set_xlim([0.1, 10])

# show color scale
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot, cax=cbar_ax, ticks=[0, -1, -2, -3])

cbar.ax.set_ylabel('fold-change')
cbar.ax.set_yticklabels(['1', '0.1', '0.01', '0.001'])
cbar.ax.tick_params(width=0) 

plt.subplots_adjust(wspace=0.3)
plt.savefig(figdir + 'noise_theory_vs_exp_mult.pdf', bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_114_0.png)


This factor of two definitely improves the predictions. Let's plot again the noise as a function of inducer concentration, but this time including this factor of 2.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
fig = plt.figure(figsize=(5, 3))
# Define outer grispec to keep at top the fold-change and at the bottom
# the noise
gs_out = mpl.gridspec.GridSpec(
    2, 1, height_ratios=[1, 1 + 1 / 5], hspace=0.1, wspace=0.05
)

# make nested gridspecs
gs_fc = mpl.gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs_out[0], wspace=0.05
)
gs_noise = mpl.gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=gs_out[1],
    wspace=0.05,
    hspace=0.01,
    height_ratios=[1, 5],
)

# Add axis to plots
# fold-change
ax_fc = [plt.subplot(gs) for gs in gs_fc]
# noise
ax_noise = [plt.subplot(gs) for gs in gs_noise]

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Plot fold-change
    # Linear
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_fc[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_fold_change,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

    # Plot noise
    # Linear
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_noise * results.params[0],
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )
    # Log
    ax_noise[op_idx[group[0]] + 3].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_noise * results.params[0],
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )

# Define data threshold
dthresh = 7
# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    # Plot fold_change
    ax_fc[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.fold_change,
        yerr=[
            data.fold_change - data.fold_change_lower,
            data.fold_change_upper - data.fold_change,
        ],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]] + 3].errorbar(
        x=data.IPTG_uM,
        y=data.noise,
        yerr=[data.noise - data.noise_lower, data.noise_upper - data.noise],
        fmt="o",
        ms=2,
        color=col_dict[group[0]][group[1]],
        label="",
    )
    # Plot noise
    ax_noise[op_idx[group[0]]].plot(
        data[data.noise > dthresh].IPTG_uM,
        data[data.noise > dthresh].noise,
        color="w",
        markeredgecolor=col_dict[group[0]][group[1]],
        label="",
        lw=0,
        marker="o",
        markersize=2,
    )

##  Set shared axis

# fold-change
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_fc[i]
    # join axis with first plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_fc[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Remove x ticks from left plot
plt.setp(ax_fc[0].get_xticklabels(), visible=False)
# Set axis to be shared with left lower plot
ax_fc[0].get_shared_x_axes().join(ax_fc[0], ax_noise[3])

# noise upper
# Loop through axis
for i in range(1, 3):
    # Select axis
    ax = ax_noise[i]
    # join x axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    # join y axis with upper left plot
    ax.get_shared_y_axes().join(ax, ax_noise[0])
    # Remove x and y ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
# Set upper left plot x axis to be shared with lower left plot
ax.get_shared_x_axes().join(ax_noise[0], ax_noise[3])
# Remove x ticks from left plot
plt.setp(ax_noise[0].get_xticklabels(), visible=False)

# noise lower
# Loop through axis
for i in range(4, 6):
    # Select axis
    ax = ax_noise[i]
    # join axis with lower left plot
    ax.get_shared_x_axes().join(ax, ax_noise[3])
    ax.get_shared_y_axes().join(ax, ax_noise[3])
    # Remove y ticks labels
    plt.setp(ax.get_yticklabels(), visible=False)

# Set scales of reference plots and the other ones will follow
ax_noise[3].set_xscale("symlog", linthreshx=thresh)  # , linscalex=0.5)
ax_noise[0].set_yscale("log")

# Set limits
for i in range(3):
    ax_fc[i].set_ylim([-0.05, 1.4])

ax_noise[0].set_ylim([dthresh, 5e2])
ax_noise[3].set_ylim([-0.25, dthresh])

# Label axis
for i, ax in enumerate(ax_fc):
    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax.set_title(label, bbox=dict(facecolor="#ffedce"))
    # Set legend
    leg = ax.legend(title="rep./cell", fontsize=5)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=5)
    leg2 = ax_noise[i + 3].legend(
        title="rep./cell", fontsize=5, loc="upper right"
    )
    plt.setp(leg2.get_title(), fontsize=5)

    ax_noise[i + 3].set_xlabel(r"IPTG ($\mu$M)")

# Set ticks for the upper noise plot
ax_noise[0].set_yticks([1e1, 1e2])
ax_noise[1].set_yticks([1e1, 1e2])
ax_noise[2].set_yticks([1e1, 1e2])

# Add y axis labels
ax_fc[0].set_ylabel(r"fold-change")
ax_noise[3].set_ylabel(r"noise")

# Align y axis labels
fig.align_ylabels()

plt.savefig(figdir + 'moment_comparison_mult_factor_lin_scale.pdf',
            bbox_inches='tight')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_116_0.png)


There is an interesting trend in which for the low noise limit this factor of two definitely improves the prediction enormously, while for high noise strains (usually hard to measure experimentally) is a more ambivalent result.

# Skewness

Another of the things we can explore to compare how our model fails to capture the experimental data is to compute the skewness. The skewness $S$ for our data is defined as
$$
S = \ee{\left( {X - \mu_X \over \sigma_X} \right)^3},
$$
where $\mu_X = \ee{X}$, and $\sigma_X = \ee{(X - \mu_X)^2}$ are the mean and the standard deviation of the variable X respectively. Expanding this and simplifying terms one can show that the skewness can be expressed in terms of moments as
$$
S = {\ee{X^3} - 3 \mu_X \sigma_X^2 - \mu_X^3 \over \sigma_X^3}.
$$
Let's apply this formula directly to our theoretical predictions for the moments.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Compute the skewness for the multi-promoter data
m_mean = df_mom_iptg.m1p0
p_mean = df_mom_iptg.m0p1
m_var = df_mom_iptg.m2p0 - df_mom_iptg.m1p0 ** 2
p_var = df_mom_iptg.m0p2 - df_mom_iptg.m0p1 ** 2

df_mom_iptg = df_mom_iptg.assign(
    m_skew=(df_mom_iptg.m3p0 - 3 * m_mean * m_var - m_mean**3)
    / m_var**(3 / 2),
    p_skew=(df_mom_iptg.m0p3 - 3 * p_mean * p_var - p_mean**3)
    / p_var**(3 / 2),
)
```

Now we are ready to compare these results with the actual experimental values. These values of the skewness were computed in the bootstrap sampling along with the noise and fold-change using `scipy`'s `skew` function.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract regulated promoter information
df_noise_reg = df_noise[df_noise.repressor > 0]
# Define repressor copy numbers to include
rep = df_noise_reg["repressor"].unique()

# Group moments by operator and repressor
df_group_exp = (
    df_noise_reg
    .sort_values("IPTG_uM")
    .groupby(["operator", "repressor"])
)

df_group = (
    df_mom_iptg[df_mom_iptg["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_skew,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # linear scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_skew,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    ax[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.skewness,
        yerr=[data.skewness - data.skewness_lower, 
        data.skewness_upper - data.skewness],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )


for i, a in enumerate(ax):
    # systematically change axis for all subplots
    ax[i].set_xscale("symlog", linthreshx=thresh, linscalex=0.5)
    # Set legend
    leg = ax[i].legend(title="rep./cell", fontsize=8)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)

    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i].set_xlabel(r"IPTG (µM)")
ax[0].set_ylabel(r"skewness")

# Change spacing between plots
plt.subplots_adjust(wspace=0.05)

plt.savefig(figdir + "skewness_comparison_exp.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_122_0.png)


The data seems to be systematically off in a similar way to the noise that we explored before. Let's explore if a multiplicative factor could as well explain this systematic deviation.

### Multiplicative factor for skewness

To check if a multiplicative factor could explain the systematic deviation with higher moments let's again plot the experimental vs theoretical skewness as we did for the noise


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Compute the skewness for the multi-promoter data
m_mean = df_mom_rep.m1p0
p_mean = df_mom_rep.m0p1
m_var = df_mom_rep.m2p0 - df_mom_rep.m1p0 ** 2
p_var = df_mom_rep.m0p2 - df_mom_rep.m0p1 ** 2

df_mom_rep = df_mom_rep.assign(
    m_skew=(df_mom_rep.m3p0 - 3 * m_mean * m_var - m_mean**3)
    / m_var**(3 / 2),
    p_skew=(df_mom_rep.m0p3 - 3 * p_mean * p_var - p_mean**3)
    / p_var**(3 / 2),
)

# Initialize list to save theoretical noise
thry_skew = list()
# Iterate through rows
for idx, row in df_noise.iterrows():
    # Extract information
    rep = float(row.repressor)
    op = row.operator
    if np.isnan(row.IPTG_uM):
        iptg = 0
    else:
        iptg = row.IPTG_uM
    
    # Extract equivalent theoretical prediction
    thry = df_mom_rep[(df_mom_rep.repressor == rep) &
                       (df_mom_rep.operator == op) &
                       (df_mom_rep.inducer_uM == iptg)].p_skew
    # Append to list
    thry_skew.append(thry.iloc[0])
    
df_noise = df_noise.assign(skew_theory = thry_skew)
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Initialize figure
fig, ax = plt.subplots(1, 2, figsize=(7, 3))

# Linear scale

# Plot reference line
ax[0].plot([1e-2, 1e2], [1e-2, 1e2], "--", color="gray")

# Plot error bars
ax[0].errorbar(
    x=df_noise.skew_theory,
    y=df_noise.skewness,
    yerr=[
        df_noise.skewness - df_noise.skewness_lower,
        df_noise.skewness_upper - df_noise.skewness,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

# Plot data with color depending on log fold-change
ax[0].scatter(
    df_noise.skew_theory,
    df_noise.skewness,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[0].set_xlabel("theoretical skewness")
ax[0].set_ylabel("experimental skewness")
ax[0].set_title("linear scale")

ax[0].set_xlim(0, 8)
ax[0].set_ylim(0, 20)

# Log scale

# Plot reference line
line = [1e-1, 1e2]
ax[1].loglog(line, line, "--", color="gray")
# Plot data with color depending on log fold-change

ax[1].errorbar(
    x=df_noise.skew_theory,
    y=df_noise.skewness,
    yerr=[
        df_noise.skewness - df_noise.skewness_lower,
        df_noise.skewness_upper - df_noise.skewness,
    ],
    color="gray",
    alpha=0.5,
    mew=0,
    zorder=0,
    fmt=".",
)

plot = ax[1].scatter(
    df_noise.skew_theory,
    df_noise.skewness,
    c=np.log10(df_noise.fold_change),
    cmap="viridis",
    s=10,
)

ax[1].set_xlabel("theoretical skewness")
ax[1].set_ylabel("experimental skewness")
ax[1].set_title("log scale")
ax[1].set_xlim([0.5, 10])

# show color scale
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plot, cax=cbar_ax, ticks=[0, -1, -2, -3])

cbar.ax.set_ylabel("fold-change")
cbar.ax.set_yticklabels(["1", "0.1", "0.01", "0.001"])
cbar.ax.tick_params(width=0)

plt.subplots_adjust(wspace=0.3)
plt.savefig(figdir + "skew_theory_vs_exp.pdf", bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_127_0.png)


There is definitely a trend, especially for the terms with lower skewness. Let's perform the weighted linear regression now to see if a multiplicative factor could explain this deviation.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract fold-change
fc = df_noise.fold_change.values
# Set values for ∆lacI to be fold-change 1
fc[np.isnan(fc)] = 1
# Normalize weights
weights = fc / fc.sum()

# Declare linear regression model
wls_model = sm.WLS(df_noise.skewness.values,
                   df_noise.skew_theory.values,
                   weights=weights)
# Fit parameter
results = wls_model.fit()
print(f'Skewness multiplicative factor: {results.params[0]}')
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>
    Skewness multiplicative factor: 2.070072211895667


Let's use this multiplicative factor and see if that indeed fixes the systematic deviation.


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> input </code>

    </div>
```python
# Extract regulated promoter information
df_noise_reg = df_noise[df_noise.repressor > 0]
# Define repressor copy numbers to include
rep = df_noise_reg["repressor"].unique()

# Group moments by operator and repressor
df_group_exp = (
    df_noise_reg
    .sort_values("IPTG_uM")
    .groupby(["operator", "repressor"])
)

df_group = (
    df_mom_iptg[df_mom_iptg["repressor"].isin(rep)]
    .sort_values("inducer_uM")
    .groupby(["operator", "repressor"])
)

# Initialize figure
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)

# Loop through groups on multi-promoter
for i, (group, data) in enumerate(df_group):
    # Log scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM >= thresh].inducer_uM,
        data[data.inducer_uM >= thresh].p_skew * 2,
        color=col_dict[group[0]][group[1]],
        label=int(group[1]),
    )
    # linear scale
    ax[op_idx[group[0]]].plot(
        data[data.inducer_uM <= thresh].inducer_uM,
        data[data.inducer_uM <= thresh].p_skew * 2,
        color=col_dict[group[0]][group[1]],
        label="",
        linestyle=":",
    )

# Loop through groups on experimental data
for i, (group, data) in enumerate(df_group_exp):
    ax[op_idx[group[0]]].errorbar(
        x=data.IPTG_uM,
        y=data.skewness,
        yerr=[data.skewness - data.skewness_lower, 
        data.skewness_upper - data.skewness],
        fmt="o",
        ms=3.5,
        color=col_dict[group[0]][group[1]],
        label="",
    )


for i, a in enumerate(ax):
    # systematically change axis for all subplots
    ax[i].set_xscale("symlog", linthreshx=thresh, linscalex=0.5)
    # Set legend
    leg = ax[i].legend(title="rep./cell", fontsize=8)
    # Set legend font size
    plt.setp(leg.get_title(), fontsize=8)

    # Set title
    label = r"$\Delta\epsilon_r$ = {:.1f} $k_BT$".format(energies[i])
    ax[i].set_title(label, bbox=dict(facecolor="#ffedce"))
    # Label axis
    ax[i].set_xlabel(r"IPTG (µM)")
ax[0].set_ylabel(r"skewness")

# Change spacing between plots
plt.subplots_adjust(wspace=0.05)

plt.savefig(figdir + "skewness_comparison_mult_factor_exp.pdf",
            bbox_inches="tight")
```


   <div style="background-color: #faf9f9; width: 100%;
               color: #a6adb5; height:15pt; margin: 0px; padding:0px;text-align: center;">

    <code style="font-size: 9pt; padding-bottom: 10px;"> output </code>

    </div>

![png](moment_dynamics_cell_division_files/moment_dynamics_cell_division_131_0.png)


Indeed the factor of two seems to resolve the issues with the skewness.
