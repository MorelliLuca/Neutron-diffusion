"""This module provides alternative ways to integrate PDE equation usign different
   types of optimizations, from `Numba` and `Scipy`.

   For performance it is best to use `nopython_integrate` for smaller grids, while `sci_integrate` becomes
   significantly faster for bigger grids (almost 2x on a 50x50 grid).

   Please, do not use `parallel_integration` since it was just a sort of experiment and it is by far the slowest.

   Functions:
   ----------
   - `parallel_dot`: implements the matrix vector multiplication expoiting parallelization of Numba,
   - `nopython_integrate`: implement numerical integration of a given PDE by compiling the fucntion in Numba _nopython_ mode,
   - `paralell_integrate`: utilizing `parallel_dot`, integrate a given PDE in `nopython` mode and with parallel matrix multiplication.
   - `sci_integrate`: exploits Scipy linear algebra methods for sparce matrices.
    """

import numpy as np
import numba as nb
from numba import jit
import scipy.sparse as sp
from scipy.sparse.linalg import inv as sparse_inv


@jit(nopython=True, parallel=True)
def parallel_dot(matrix: np.ndarray, vector: np.ndarray):
    r"""Implementation of vector matrix product with parallelization, using Numba.

    Given $M_{ij}$ and $V_{i}$, returns $$W_{i}=\sum_{j} A_{ij} V_j.$$


    Parameters
    ----------
    `matrix` : np.ndarray
        2D array, which the first factor of the product.
    vector : _type_
        1D array, which is the second factor of the product.

    Returns
    -------
    np.ndarray
        Result 1D array (vector).
    """
    assert matrix.shape[1] == vector.shape[0]
    result = np.zeros(matrix.shape[0], dtype=matrix.dtype)
    for i in nb.prange(matrix.shape[0]):
        for j in nb.prange(matrix.shape[1]):
            result[i] += matrix[i, j] * vector[j]
    return result


@jit(nopython=True, cache=True, fastmath=True)
def nopython_integrate(
    phi: np.ndarray,
    PDE_matrix: np.ndarray,
    sources: np.ndarray,
    omega: float,
    conv_criterion: float,
    empty_cells: np.ndarray,
    grid_size: np.ndarray,
) -> np.ndarray:
    r"""Performs numerical integration of a given statuonary PDE using Numba _nopython_ compliation, for better performances.

    The solution in obatined using successive relaxation method:
    1. first, the PDE_matrix is decomposed into its lower tringular component $L$
    and the upper one $-U$
    2. iterating, from an initial guess,
    $$\vec \phi_{n}= L^{-1}U \vec \phi_{n-1}+L^{-1} \vec S,$$ where $\vec S$ is
    `sources`, we obtain approximations of the solution
    3. a better approxiamtion is reached by weighting $\phi_n'= \phi_n\omega+ (1-\omega)\phi_{n-1}$
    4. when $ \|\phi_{n}-\phi_{n-1}|<$ `conv_criterion` convergence is met.

    Overall, this tested to be the best integration method performace-wise.

    Parameters
    ----------
    phi : np.ndarray
        1D array containing the initial guess of the solution.
    PDE_matrix : np.ndarray
        2D array describing the PDE.
    sources : np.ndarray
        1D array representing the sources terms of the PDE.
    omega : float
        Weight between the previous iteration solution and the solution at the new iteration.
    conv_criterion : float
        Criterion of convergence: if the norm of two consecuitive solutions (vectors)
        is smaller that this parameter convergence is met.
    empty_cells : np.ndarray
        1D array listing positions in the grid of simulation (tuples) of the empty cells.
    grid_size : np.ndarray
        1D array containing the size (rows, columns) of the grid of over which we integrate.

    Returns
    -------
    np.ndarray
        1D array containing the converged solution.
    """
    L = np.tril(PDE_matrix)  # Lower triangular PDE Matrix
    U = -1 * np.triu(PDE_matrix, 1)  # Upper triangular PDE Matrix times -1
    for i in nb.prange(np.prod(grid_size)):
        if L[i, i] == 0:
            L[i, i] = 1
    L_inv = np.linalg.inv(L)
    # Iterations to find the solution
    # Stops when succesive solutions start to be close (their norm)
    not_converged = True
    while not_converged:
        old_phi = phi.copy()
        tmp_phi = L_inv.dot(U.dot(phi)) + L_inv.dot(sources)
        # Sets empty cell flux to zero
        for cell_coord in empty_cells:
            index = cell_coord[0] + cell_coord[1] * grid_size[1]
            if index > 0:
                tmp_phi[index] = 0
        phi = omega * tmp_phi + (1 - omega) * old_phi
        # Convergence criterion
        if np.linalg.norm(tmp_phi - old_phi) < conv_criterion:
            not_converged = False
    return phi


@jit(nopython=True, cache=False, parallel=False)
def paralell_integrate(
    phi: np.ndarray,
    PDE_matrix: np.ndarray,
    sources: np.ndarray,
    omega: float,
    conv_criterion: float,
    empty_cells: np.ndarray,
    grid_size: np.ndarray,
) -> np.ndarray:
    r"""Performs numerical integration of a given statuonary PDE using Numba _nopython_ compliation
    and parallelization of matrix multiplications, for better performances.

    The solution in obatined using successive relaxation method:
    1. first, the PDE_matrix is decomposed into its lower tringular component $L$
    and the upper one $-U$
    2. iterating, from an initial guess,
    $$\vec \phi_{n}= L^{-1}U \vec \phi_{n-1}+L^{-1} \vec S,$$ where $\vec S$ is
    `sources`, we obtain approximations of the solution
    3. a better approxiamtion is reached by weighting $\phi_n'= \phi_n\omega+ (1-\omega)\phi_{n-1}$
    4. when $ \|\phi_{n}-\phi_{n-1}|<$ `conv_criterion` convergence is met.

    Parameters
    ----------
    phi : np.ndarray
        1D array containing the initial guess of the solution.
    PDE_matrix : np.ndarray
        2D array describing the PDE.
    sources : np.ndarray
        1D array representing the sources terms of the PDE.
    omega : float
        Weight between the previous iteration solution and the solution at the new iteration.
    conv_criterion : float
        Criterion of convergence: if the norm of two consecuitive solutions (vectors)
        is smaller that this parameter convergence is met.
    empty_cells : np.ndarray
        1D array listing positions in the grid of simulation (tuples) of the empty cells.
    grid_size : np.ndarray
        1D array containing the size (rows, columns) of the grid of over which we integrate.

    Returns
    -------
    np.ndarray
        1D array containing the converged solution.
    """
    L = np.tril(PDE_matrix)  # Lower triangular PDE Matrix
    U = -1 * np.triu(PDE_matrix, 1)  # Upper triangular PDE Matrix times -1
    for i in nb.prange(np.prod(grid_size)):
        if L[i, i] == 0:
            L[i, i] = 1
    L_inv = np.linalg.inv(L)
    # Iterations to find the solution
    # Stops when succesive solutions start to be close (their norm)
    not_converged = True
    while not_converged:
        old_phi = phi.copy()
        tmp_phi = parallel_dot(L_inv, parallel_dot(U, phi)) + parallel_dot(
            L_inv, sources
        )
        # Sets empty cell flux to zero
        for cell_coord in empty_cells:
            index = cell_coord[0] + cell_coord[1] * grid_size[1]
            if index > 0:
                tmp_phi[index] = 0
        phi = omega * tmp_phi + (1 - omega) * old_phi
        # Convergence criterion
        if np.linalg.norm(tmp_phi - old_phi) < conv_criterion:
            not_converged = False
    return phi


def sci_integrate(
    phi: np.ndarray,
    tmp_PDE_matrix: np.ndarray,
    sources: np.ndarray,
    omega: float,
    conv_criterion: float,
    empty_cells: np.ndarray,
    grid_size: np.ndarray,
) -> np.ndarray:
    r"""Performs numerical integration of a given statuonary PDE using Scipy sparce matrix methods, for better performances.

    The solution in obatined using successive relaxation method:
    1. first, the PDE_matrix is decomposed into its lower tringular component $L$
    and the upper one $-U$
    2. iterating, from an initial guess,
    $$\vec \phi_{n}= L^{-1}U \vec \phi_{n-1}+L^{-1} \vec S,$$ where $\vec S$ is
    `sources`, we obtain approximations of the solution
    3. a better approxiamtion is reached by weighting $\phi_n'= \phi_n\omega+ (1-\omega)\phi_{n-1}$
    4. when $ \|\phi_{n}-\phi_{n-1}|<$ `conv_criterion` convergence is met.

    For larger matrixes this implementation can be faster than `nopython_integrate`.

    Parameters
    ----------
    phi : np.ndarray
        1D array containing the initial guess of the solution.
    PDE_matrix : np.ndarray
        2D array describing the PDE.
    sources : np.ndarray
        1D array representing the sources terms of the PDE.
    omega : float
        Weight between the previous iteration solution and the solution at the new iteration.
    conv_criterion : float
        Criterion of convergence: if the norm of two consecuitive solutions (vectors)
        is smaller that this parameter convergence is met.
    empty_cells : np.ndarray
        1D array listing positions in the grid of simulation (tuples) of the empty cells.
    grid_size : np.ndarray
        1D array containing the size (rows, columns) of the grid of over which we integrate.

    Returns
    -------
    np.ndarray
        1D array containing the converged solution.
    """
    for i in nb.prange(np.prod(grid_size)):
        if tmp_PDE_matrix[i, i] == 0:
            tmp_PDE_matrix[i, i] = 1
    PDE_matrix = sp.csc_matrix(tmp_PDE_matrix)
    L = sp.tril(PDE_matrix, format="csc")  # Lower triangular PDE Matrix
    U = -1 * sp.triu(
        PDE_matrix, 1, format="csc"
    )  # Upper triangular PDE Matrix times -1
    L_inv = sparse_inv(L)
    # Iterations to find the solution
    # Stops when succesive solutions start to be close (their norm)
    not_converged = True
    while not_converged:
        old_phi = phi.copy()
        tmp_phi = L_inv.dot(U.dot(phi)) + L_inv.dot(sources)
        # Sets empty cell flux to zero
        for cell_coord in empty_cells:
            index = cell_coord[0] + cell_coord[1] * grid_size[1]
            if index > 0:
                tmp_phi[index] = 0
        phi = omega * tmp_phi + (1 - omega) * old_phi
        # Convergence criterion
        if np.linalg.norm(tmp_phi - old_phi) < conv_criterion:
            not_converged = False
    return phi
