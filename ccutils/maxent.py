# -*- coding: utf-8 -*-
"""
Title:
    maxent.py
Last update:
    2018-11-22
Author(s):
    Manuel Razo-Mejia
Purpose:
    This file compiles the functions necessary to compute the maxEnt
    approximation of the distribution relevant to the channel capacity
    project.
"""

import numpy as np
import pandas as pd
import scipy as sp

# Import library to perform maximum entropy fits
from maxentropy.skmaxent import FeatureTransformer, MinDivergenceModel


# Function used with the maxentropy package to fit the Lagrange multipliers of
# the MaxEnt distribution
def feature_fn(x, x_expo):
    '''
    For a given mRNA protein pair and given exponents it computes
    the product of mRNA**mRNA_expo * protein**protein_expo.
    '''
    return x[0]**x_expo[0] * x[1]**x_expo[1]


def MaxEnt_bretthorst(constraints, features,
                      algorithm='BFGS', tol=1E-4, paramtol=5E-5, maxiter=1000):
    '''
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
    '''
    # Define a dummy samplespace that we don't need since we are giving the
    # matrix of pre-computed features, but the maxentropy package still
    # requires it.
    samplespace = np.zeros(np.max(features.shape))

    # # First rescaling # #

    # Compute the factor to be used to re-scale the problem
    rescale_factor = np.sqrt(np.sum(features**2, axis=1))

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
    model.callingback = True  # TBH I don't know what this does but it is needed
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


def maxEnt_from_lagrange(mRNA, protein, lagrange,
                         exponents=[(1, 0), (2, 0), (3, 0),
                                    (0, 1), (0, 2), (1, 1)], log=False):
    '''
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
    '''
    # Generate grid of points
    mm, pp = np.meshgrid(mRNA, protein)

    # Initialize 3D array to save operations associated with each lagrange
    # multiplier
    operations = np.zeros([len(lagrange), len(protein), len(mRNA)])

    # Compute operations associated with each Lagrange Multiplier
    for i, expo in enumerate(exponents):
        operations[i, :, :] = lagrange[i] * mm**expo[0] * pp**expo[1]

    # check if the log probability should be returned
    if log:
        return np.sum(operations, axis=0) -\
            sp.misc.logsumexp(np.sum(operations, axis=0))
    else:
        return np.exp(np.sum(operations, axis=0) -
                      sp.misc.logsumexp(np.sum(operations, axis=0)))

