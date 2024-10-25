"""This module provides a set of classes that can be utilized to perform numerical integration of stationary PDEs.
   
These classes create a discretized configuartion space for the PDE, then using succesive relaxation method, they estimate the 
stationary solution to the given PDE.

The class `Cell` represent the single spacial element of the discretization, containing its numerical proprieties such as the position,
then the `Grid` class groups toghether all the `Cell` istances and gives to the user the methods needed to represent the discretized PDE.
Lastly, the class `Solver` manages the integration of the PDE.
"""

import numpy as np
from copy import deepcopy
from .Functions import *
from .IntegrationsMethods import *

# ---- Classes -----


class Cell:
    """Represents a single element (cell) in the discretized reactor configuration space.
       This class is used to keep track of all the discretized variables that depends on space positions.

    Attributes
    ----------
    `flux` : float
        Neutron flux in the cell.
    `position` : tuple
        Coordinates of the cell in a grid.
    `type` : str={"Flux", "Empty"}
        Type of the cell, e.g. whether it is empty or not.
    """

    def __init__(self, flux: float, position: tuple, type: str):
        """Initiates a cell in specified `position` and with specified `flux` and `type`

        Parameters
        ----------
        `flux` : float
            Neutron flux of the cell.
        `position` : tuple
            Coordinates of the cell in a grid.
        `type` : str={"Flux", "Empty"}
            Type of the cell, e.g. whether it is empty or not.
        """
        self.flux = flux
        self.position = position
        self.type = type


class Grid:
    """Represents the entire grid used for spatial discretization of the reactor.

    Attributes
    ----------
    `Delta` : float
        The discretization step: size of each cell (all equal size).
    `size` : np.ndarray
        The maximum number of cells in the grid: (max_x_size, ,max_y_size).
    `cells_list` : np.ndarray
        The list of cells of the grid.
    `empty_cells` : np.ndarray
        The list of  the coordinates of cells of type `Empty` in the grid.

    Methods
    -------
    * `flux_vector()` :
        Returns a vector containing the fluxes of all the cells.
    * `flux_matrix()` :
        Returns a matrix containing the fluxes of all the cells.
    * `first_Xderivative_matrix()` :
        Returns the matrix of the discretized partial derivative along the $x$-axis.
    * `first_Yderivative_matrix()` :
        Returns the matrix of the discretized partial derivative along the $y$-axis.
    * `second_Xderivative_matrix()` :
        Returns the matrix of the discretized partial second derivative along the $x$-axis.
    * `second_Yderivative_matrix()` :
        Returns the matrix of the discretized partial second derivative along the $y$-axis.

    """

    def __init__(self, grid_matrix: list, Delta: float):
        """Initializes the grid with the specified size and cells.

        Parameters
        ----------
        `grid_matrix` : list
            A 2-D list describing the geometry of the reactor. The allowed values are:
            - float number for cells in which simulate the reactor (the number is the flux),
            - "E" for cells that are not part of the reactor (they will always have $0$ flux).
        `Delta` : float
            The discretization step (grid spacing).
        """
        self.Delta = Delta
        row_size = len(grid_matrix[0])
        column_size = len(grid_matrix)
        for row in grid_matrix:
            if row_size != len(row):
                raise Exception("Grid matrix is not uniform.")
        self.size = np.array((row_size, column_size), dtype=np.int64)
        tmp_cells_list = []
        tmp_empty_cells = []
        # Gives type and flux to each cell from grid_matrix values
        for row_index in range(self.size[1]):
            for column_index in range(self.size[0]):
                if isinstance(
                    grid_matrix[row_index][column_index], float
                ) or isinstance(grid_matrix[row_index][column_index], int):
                    tmp_cells_list.append(
                        Cell(
                            grid_matrix[row_index][column_index],
                            (row_index, column_index),
                            "Flux",
                        )
                    )
                elif grid_matrix[row_index][column_index] == "E":
                    tmp_cells_list.append(Cell(0, (row_index, column_index), "Empty"))
                    tmp_empty_cells.append((row_index, column_index))
                else:
                    raise ValueError(
                        "Cell ("
                        + str(row_index)
                        + ", "
                        + str(column_index)
                        + ") type not valid"
                    )
                self.cells_list = np.array(tmp_cells_list)
                if len(tmp_empty_cells) == 0:
                    self.empty_cells = np.array([(-1, -1)], dtype=np.int64)
                else:
                    self.empty_cells = np.array(tmp_empty_cells, dtype=np.int64)

    def flux_vector(self) -> np.ndarray:
        """Returns a vector containing the fluxes of all the cells:
        the grid is read row by row and then each row is attached one after the other to form the vector.

        Returns
        -------
        np.ndarray
           1D array that lists the fluxes of all the cells.

        Notes
        -----
        This vector representation is used by `Solver.solve()` for numerically integration.
        """
        vector = np.empty(np.prod(self.size), dtype=np.float64)
        for cell in self.cells_list:
            if cell.type != "Empty":
                vector[cell.position[0] + cell.position[1] * self.size[1]] = cell.flux
            else:
                assert cell.flux == 0
                vector[cell.position[0] + cell.position[1] * self.size[1]] = cell.flux
        return vector

    def flux_matrix(self) -> np.ndarray:
        """Returns a matrix containing the fluxes of all the cells:
        the $(i, j)$ element of the matrix represents the cell in the $i$-th column and $j$-th row.

        Returns
        -------
        np.ndarray
           2D array that lists the fluxes of all the cells.

        Notes
        -----
        This matrix is usually fed to matplotlib to get a graphical representation of the reactor.
        """
        matrix = np.empty(self.size, dtype=np.float64)
        for cell in self.cells_list:
            if cell.type != "Empty":
                matrix[cell.position[1], cell.position[0]] = cell.flux
            else:
                assert cell.flux == 0
                matrix[cell.position[1], cell.position[0]] = cell.flux
        return matrix

    def flux_PDE_matrix(self) -> np.ndarray:
        """Returns a matrix representing a linear flux term in the PDE.

        Returns
        -------
        np.ndarray
           2D array that is used to represent linear terms in the PDE.

        See also
        -----
        `Solver` : The attribute `Solver.PDE_matrix` can be created usign this method:
        """
        matrix = np.empty((np.prod(self.size), np.prod(self.size)), dtype=np.float64)
        for cell in self.cells_list:
            row_index = cell.position[0] + cell.position[1] * self.size[1]
            column_index = row_index
            if cell.type == "Flux":
                matrix[column_index, row_index] = 1
            else:
                assert cell.flux == 0
                matrix[cell.position[1], cell.position[0]] = cell.flux
        return matrix

    def first_Xderivative_matrix(self) -> np.ndarray:
        r"""Returns a matrix representing the first partial derivative along the $x$-axsis:
        after matrix multiplication with the vector returned by `flux_vector()`, this matrix allows
        to obtain the vector approximating the derivative by the central_ difference_ method.
        $$\partial_x \phi\big|_{i,j} \approx \frac{\phi_{i+1,j} - \phi_{i-1,j}}{2\Delta},$$
        where $\phi_{i,j}$ is the fulx in the cell (i, j).

        Returns
        -------
        np.ndarray
           2D array that is used to evaluate the partial derivative along the $x$-axis.

        See also
        -----
        `Solver` : The attribute `Solver.PDE_matrix` can be created usign this method:
        """
        matrix = np.empty((np.prod(self.size), np.prod(self.size)), dtype=np.float64)
        for row in self.cells_list:
            row_index = row.position[0] + row.position[1] * self.size[1]
            for column in self.cells_list:
                column_index = column.position[0] + column.position[1] * self.size[1]
                if column.type == "Flux" and row.type == "Flux":
                    if column.position == tuple(np.subtract(row.position, (1, 0))):
                        matrix[column_index, row_index] = -1 / (2 * self.Delta)
                    elif column.position == tuple(np.add(row.position, (1, 0))):
                        matrix[column_index, row_index] = 1 / (2 * self.Delta)
                    else:
                        matrix[column_index, row_index] = 0
                else:
                    matrix[column_index, row_index] = 0
        return matrix

    def first_Yderivative_matrix(self) -> np.ndarray:
        r"""Returns a matrix representing the first partial derivative along the $y$-axsis:
        after matrix multiplication with the vector returned by `flux_vector()`, this matrix allows
        to obtain the vector approximating the derivative by the central_ difference_ method.
        $$\partial_y \phi\big|_{i,j} \approx \frac{\phi_{i,j+1} - \phi_{i,j-1}}{2\Delta},$$
        where $\phi_{i,j}$ is the fulx in the cell (i, j).

        Returns
        -------
        np.ndarray
           2D array that is used to evaluate the partial derivative along the $y$-axis.

        See also
        -----
        `Solver` : The attribute `Solver.PDE_matrix` can be created usign this method:
        """
        matrix = np.empty((np.prod(self.size), np.prod(self.size)), dtype=np.float64)
        for row in self.cells_list:
            row_index = row.position[0] + row.position[1] * self.size[1]
            for column in self.cells_list:
                column_index = column.position[0] + column.position[1] * self.size[1]
                if column.type == "Flux" and row.type == "Flux":
                    if column.position == tuple(np.subtract(row.position, (0, 1))):
                        matrix[column_index, row_index] = -1 / (2 * self.Delta)
                    elif column.position == tuple(np.add(row.position, (0, 1))):
                        matrix[column_index, row_index] = 1 / (2 * self.Delta)
                    else:
                        matrix[column_index, row_index] = 0
                else:
                    matrix[column_index, row_index] = 0
        return matrix

    def second_Xderivative_matrix(self) -> np.ndarray:
        r"""Returns a matrix representing the second partial derivative along the $x$-axsis:
        after matrix multiplication with the vector returned by `flux_vector()`, this matrix allows
        to obtain the vector approximating the derivative by the central_ difference_ method:
        $$\partial^2_x \phi\big|_{i,j} \approx \frac{\phi_{i+1,j} -2\phi_{i,j}+ \phi_{i-1,j}}{\Delta^2},
        $$ where $\phi_{i,j}$ is the fulx in the cell (i, j).

        Returns
        -------
        np.ndarray
           2D array that is used to evaluate the second partial derivative along the $x$-axis.

        See also
        -----
        `Solver` : The attribute `Solver.PDE_matrix` can be created usign this method.
        """
        matrix = np.empty((np.prod(self.size), np.prod(self.size)), dtype=np.float64)
        for row in self.cells_list:
            row_index = row.position[0] + row.position[1] * self.size[1]
            for column in self.cells_list:
                column_index = column.position[0] + column.position[1] * self.size[1]
                if column.type == "Flux" and row.type == "Flux":
                    if column.position == tuple(np.subtract(row.position, (1, 0))):
                        matrix[column_index, row_index] = 1 / (self.Delta * self.Delta)
                    elif column.position == tuple(np.add(row.position, (1, 0))):
                        matrix[column_index, row_index] = 1 / (self.Delta * self.Delta)
                    elif column.position == row.position:
                        matrix[column_index, row_index] = -2 / (self.Delta * self.Delta)
                    else:
                        matrix[column_index, row_index] = 0
                else:
                    matrix[column_index, row_index] = 0
        return matrix

    def second_Yderivative_matrix(self) -> np.ndarray:
        r"""Returns a matrix representing the second partial derivative along the $y$-axsis:
        after matrix multiplication with the vector returned by `flux_vector()`, this matrix allows
        to obtain the vector approximating the derivative by the central_ difference_ method:
        $$\partial^2_y \phi\big|_{i,j} \approx \frac{\phi_{i,j+1} -2\phi_{i,j}+ \phi_{i,j-1}}{\Delta^2},
        $$ where $\phi_{i,j}$ is the fulx in the cell (i, j).

        Returns
        -------
        np.ndarray
           2D array that is used to evaluate the second partial derivative along the $y$-axis.

        See also
        -----
        `Solver` : The attribute `Solver.PDE_matrix` can be created usign this method.
        """
        matrix = np.empty((np.prod(self.size), np.prod(self.size)), dtype=np.float64)
        for row in self.cells_list:
            row_index = row.position[0] + row.position[1] * self.size[1]
            for column in self.cells_list:
                column_index = column.position[0] + column.position[1] * self.size[1]
                if column.type == "Flux" and row.type == "Flux":
                    if column.position == tuple(np.subtract(row.position, (0, 1))):
                        matrix[column_index, row_index] = 1 / (self.Delta * self.Delta)
                    elif column.position == tuple(np.add(row.position, (0, 1))):
                        matrix[column_index, row_index] = 1 / (self.Delta * self.Delta)
                    elif column.position == row.position:
                        matrix[column_index, row_index] = -2 / (self.Delta * self.Delta)
                    else:
                        matrix[column_index, row_index] = 0
                else:
                    matrix[column_index, row_index] = 0
        return matrix


class Solver:
    """
    Manages the numerical integration of a given stationary PDE.

    Attributes
    -----------
    `grid` : Grid
        Discretization of the coordinate space.
    `PDE_matrix` : np.array
        2D array that represents the discretized stationary PDE to be solved.
    `sources` : np.array
        2D array that contains the source terms of the PDE.

    Methods
    -------
    * `solve()` :
        Numerically finds the best approximation of the stationary solution of a given PDE.

    """

    def __init__(self, grid: Grid, sources: np.ndarray, PDE_matrix: np.ndarray):
        """Initializes the solver by defining the coordinate space and the PDE with its sources terms.

        Parameters
        ----------
        `grid` : Grid
            Discretization of the coordinate space.
        `PDE_matrix` : np.array
            2D array that represents the discretized stationary PDE to be solved.
        `sources` : np.array
            2D array that contains the source terms of the PDE: these are arranged
            in the array following the
            geometry of the reactor.

        See also
        --------
        `Grid.first_Xderivative_matrix()`, `Grid.first_Yderivative_matrix()`,
        `Grid.second_Xderivative_matrix()`
        and `Grid.second_Yderivative_matrix()`:
            These methods can generate the matrices that made up the matrix `PDE_matrix`
        """
        self.grid = grid
        self.PDE_matrix = PDE_matrix
        self.sources = sources

    def solve(
        self,
        omega: float,
        conv_criterion: float,
        update: bool = False,
        mode: str = "nopython",
    ) -> Grid:
        r"""Approximates the stationary flux solution.

        The solution in obatined using successive relaxation method:
        1. first, the PDE_matrix is decomposed into its lower tringular component $L$
        and the upper one $-U$
        2. iterating, from an initial guess,
        $$\vec \phi_{n}= L^{-1}U \vec \phi_{n-1}+L^{-1} \vec S,$$ where $\vec S$ is
        `sources`, we obtain approximations of the solution
        3. a better approxiamtion is reached by weighting $\phi_n'= \phi_n\omega+ (1-\omega)\phi_{n-1}$
        4. when $ \|\phi_{n}-\phi_{n-1}|<$ `conv_criterion` convergence is met.

        Using the paramenter `mode`, different optimization options can be selected.
        For more informations about these optimization consult `Reactpy.IntegrationsMethods`.

        Parameters
        ----------
        omega : float
            Weight between the previous iteration solution and the solution at the new iteration.
        conv_criterion : float
            Criterion of convergence: if the norm of two consecuitive solutions (vectors)
            is smaller that this parameter convergence is met.
        update : bool
            If True the attribute `grid` is updated with the solution, otherwise only the
            retuned Grid object contains the solution.
            (By default this is False).
        mode : string
            Determine the integration mode:
            - "nopython" (default) leaves the integration to `Numba`,
            - "parallel" parallelize the matrix multiplication with `Numba`,
            - "Scipy" uses the `Scipy` libraries to mange the matrices as sparse matrices.

        Returns
        -------
        Grid
            New Grid object containing the solution obtained.
        """
        phi = self.grid.flux_vector()  # Initial conditions
        sources_vector = matrix_to_vector(self.sources)

        if mode == "nopython":
            phi = nopython_integrate(
                phi,
                self.PDE_matrix,
                sources_vector,
                omega,
                conv_criterion,
                self.grid.empty_cells,
                self.grid.size,
            )
        elif mode == "parallel":
            phi = paralell_integrate(
                phi,
                self.PDE_matrix,
                sources_vector,
                omega,
                conv_criterion,
                self.grid.empty_cells,
                self.grid.size,
            )
        elif mode == "Scipy":
            phi = sci_integrate(
                phi,
                self.PDE_matrix,
                sources_vector,
                omega,
                conv_criterion,
                self.grid.empty_cells,
                self.grid.size,
            )
        else:
            print(
                mode
                + "not recognized.\n By default nopython integration will be executed."
            )
            phi = nopython_integrate(
                phi,
                self.PDE_matrix,
                sources_vector,
                omega,
                conv_criterion,
                self.grid.empty_cells,
                self.grid.size,
            )

        # Create the Grid of the solution
        new_grid = deepcopy(self.grid)
        for cell in new_grid.cells_list:
            cell.flux = phi[cell.position[0] + cell.position[1] * self.grid.size[1]]
        if update == True:
            self.grid = new_grid
        return new_grid

    # pdoc --docformat numpy --math Reactpy -o ../Documentation/ --no-include-undocumented
