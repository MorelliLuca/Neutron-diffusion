r"""This module provvides some functions that can generate the time evolution of the control rods.

    Each function returns a float number in $\[0,1\]$ which must then be multiplied 
    by the matrix representing the control rods disposition.

    Functions
    ---------
    - `linear_cycle`: returns value of of control bars at given time, throught a linear cycle.
    - `shutdown`: returns value of of control bars at given time: it is simualted a scenario in which all control rods
    are fully inserted instanenusely.
    """
import numpy as np

def linear_cycle(t: float, delta_t: float, t_max: float, k: float) -> float:
    r"""Reproduces the percentage of insertion of control rods at a given time `t`.
    The full time evolution is splitted in $5$ time periods in which the values returned 
    changes linearly or stay constant:
    - for `t` * `delta_t` < 10% `t_max`}  the the level of the rods is lowered from $100\%$ to $0\%$,
    - for 10% `t_max` < `t` * `delta_t` < 20% `t_max` all the rods are extracted $(0\%)$,
    - for 20% `t_max` < `t` * `delta_t` < 30% `t_max` the rods are progressively inserted until $k$ level is reached,
    - for 30% `t_max` < `t` * `delta_t` < 80% `t_max` the rods are kept at fixed insertion value $k$,
    - for 80% `t_max` < `t` * `delta_t` < `t_max` all the rods are fully inserted.

    Parameters
    ----------
    `t` : float
        Time (iteration) at which the rod insertion level is returned.
    `delta_t` : float
        Time step between iterations.
    `t_max` : float
        time duration of the cycle. (This is actual time in seconds) 
    `k` : float
        Insertion level kept during steady state operation of the reactor.

    Returns
    -------
    float
        Insertion level of the control rods a time `t`.
    """
    if t * delta_t < t_max * 0.1:
        percentage = -10 * t / t_max * delta_t + 1
    elif t * delta_t < t_max * 0.2:
        percentage = 0
    elif t * delta_t < t_max * 0.3:
        percentage = (t / t_max * delta_t - 0.2) * k * 10
    elif t * delta_t < t_max * 0.8:
        percentage = k
    else:
        percentage = (1 - k) * 5 * (t / t_max * delta_t - 0.8) + k
    return percentage


def shutdown(t: float, delta_t: float, t_max: float, k: float) -> float:
    """Reproduces the percentage of insertion of control rods at a given time `t`.
    The full time evolution is splitted in $2$ time periods in which the values returned 
    changes linearly or stay constant:
    - for the first half of `t_max` all the rods are inserted for percentage `k`,
    - all the control rods are immediately fully inserted.

    Parameters
    ----------
    `t` : float
        Time (iteration) at which the rod insertion level is returned.
    `delta_t` : float
        Time step between iterations.
    `t_max` : float
        time duration of the cycle. (This is actual time in seconds) 
    `k` : float
        Insertion level kept during steady state operation of the reactor.

    Returns
    -------
    float
        Insertion level of the control rods a time `t`.
    """
    if t * delta_t < t_max / 2:
        percentage = k
    else:
        percentage = 1
    return percentage

def linear_cycle_array(delta_t: float, t_max: float, k: float) -> np.ndarray:
    """
    Returns an array containing the values, at different iterations steps, of the intertion level of the control rods,
     following the `linear_cycle` scheme. 

    Parameters
    ----------
    `delta_t` : float
        Time step between iterations.
    `t_max` : float
        time duration of the cycle. (This is actual time in seconds) 
    `k` : float
        Insertion level kept during steady state operation of the reactor.


    Returns
    -------
    np.ndarray
        Array containg squential values over time of the insterion level of the control rods. 
    """
    array = np.zeros(int(t_max/delta_t), dtype=np.float64)
    for t in range(int(t_max/delta_t)):
        array[t] = linear_cycle(t,delta_t,t_max,k)
    return array
        
def shutdown_array(delta_t: float, t_max: float, k: float) -> np.ndarray:
    """
    Returns an array containing the values, at different iterations steps, of the intertion level of the control rods,
    following the `shutdown` shceme.

    Parameters
    ----------
    `delta_t` : float
        Time step between iterations.
    `t_max` : float
        time duration of the cycle. (This is actual time in seconds) 
    `k` : float
        Insertion level kept during steady state operation of the reactor.


    Returns
    -------
    np.ndarray
        Array containg squential values over time of the insterion level of the control rods. 
    """
    array = np.zeros(int(t_max/delta_t), dtype=np.float64)
    for t in range(int(t_max/delta_t)):
        array[t] = shutdown(t,delta_t,t_max,k)
    return array
        