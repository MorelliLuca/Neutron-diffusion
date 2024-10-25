import numpy as np

"""This module provides some recurring functions used by the other modules, but that can be useful also 
    for the user. Mainly, these functions allows to read data from files and convert it directly in Numpy 
    arrays or to to perform conversion between vectorial and matricial representation of data.
    """


def vector_to_matrix(vector: np.ndarray, size: tuple) -> np.ndarray:
    """
    Converts a vector into matrix of specified size.

    Given the tuple $(n, m)$, every n components of the `vector` are turned into a row:
    in this way the matrix element $(i, j)$ corresponds to the `vector` component $i+j*m$.

    Parameters
    ----------
    `vector` : np.ndarray
        1-D representation of the matricial data.
    `size` : tuple
        Number of (row, column) of the matrix.

    Returns
    -------
    np.ndarray
        Matricial representation of the data contained into `vector`.

    """
    matrix = np.empty(size, dtype="float64")
    for i in range(size[0]):
        for j in range(size[1]):
            matrix[i, j] = vector[i + size[1] * j]
    return matrix


def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Converts a matrix of specified size into a vector.

    All the rows become just one vector, from top to bottom:
    in this way the `matrix` element $(i, j)$ corresponds to the vector component $i+j*m$.

    Parameters
    ----------
    `matrix` : np.ndarray
        2-D representation of the vectorial data.

    Returns
    -------
    np.ndarray
        Vectorial representation of the data contained into `matrix`.

    """
    size = matrix.shape
    vector = np.empty(np.prod(size), dtype="float64")
    for i in range(size[0]):
        for j in range(size[1]):
            vector[i + size[1] * j] = matrix[i, j]
    return vector


def file_read_as_matrix(filename: str) -> list:
    """Reads from file a matrix and returns a 2-D list containing its entries.

    The matrix can contain both letters and numbers (interger, float).

    Parameters
    ----------
    filename : str
        Path to the file containing the data.

    Returns
    -------
    list
        2-D list that represents the matrix.
    """
    input_file = open(filename, "r")
    matrix = []
    for line in input_file:
        row = []
        line = line.replace("\n", "")
        for entry in line.split(" "):
            if entry.isalpha():
                row.append(entry)
            else:
                row.append(float(entry))
        matrix.append(row)
    return matrix


def file_read_as_vector(filename: str) -> list:
    """Reads from file a vector and returns a 1-D list containing its entries.

    The vector can contain both letters and numbers (interger, float).

    If the file contains more rows, those are attached one after the other.

    Parameters
    ----------
    filename : str
        Path to the file containing the data.

    Returns
    -------
    list
        1-D list that represents the matrix.
    """
    input_file = open(filename, "r")
    vector = []
    for line in input_file:
        line = line.replace("\n", "")
        for entry in line.split(" "):
            if entry.isalpha():
                vector.append(entry)
            else:
                vector.append(float(entry))
    return vector
