# Numerical neutron diffusion in nuclear reactors
This code allows to numerically integrate the time evolution of the neutron diffusion in a 2D nuclear reactor. There are three main parts:
- the first just integrates numerically the neutron dynamics of a _1 speed reactor_,
- the second integrates numerically the neutron dynamics of a _1 speed reactor_ with dynamical control rods insertion,
- the second evaluate the _fudge factor_ of a given geometry.

For this code it has also been created a package `Reactpy`, that manages the numerical integration of the system. The documentation for this package can be found in the directory `Documentation`. The _Reactpy.html_ file allows to navigate the whole documentation.

The folder `Source` contains the code that runs the integration, while the directory `Output examples` contains some examples of the simulation.
## Requirements
The project is written in python and requires the following packages:
1. Numpy
2. Numba
3. Scipy
4. Matplotlib
5. tqdm

## Usage
To use this code it is necessary to first input the data that describes the reactor geometry and material characteristics, this can be done through 5 `.dat` files.
- `Parameters.dat` specifies all the scalar (space-indiependent) variables,
- `grid.dat` contains the geometrical information of the discretization of the reactor: each cell of the discretization is represented by an entry of a matrix, if it is inserted a number this will be used as initial condition of for the neutron flux, while the letter _E_ is used to represent an empty cell (which flux is always zero).
- `Sigma_absorption.dat` contains the spacial values of $\Sigma_a$ (which quantifies how many neutrons are absorbed by the moderator) in each cell.
- `Sigma_fuel.dat` contains the spacial values of $\Sigma_f$ (which quantifies how many neutrons are produced by each external neutron-atom scattering) in each cell.
- `Control_rods.dat` specifies where the control rods are placed and their absorption value.

The code of `1SpeedReactor.py` and of `1SpeedReactor.py` then allow to numerically integrate the following equation:

$$\frac{1}{v}\frac{\partial \phi}{\partial t}=D\nabla^2\phi-(\Sigma_a+\Sigma_f)\phi+v\frac{\Sigma_f}{k_{fudge}}\phi.\$$

After the integration the code produces an animation that shows the time evolution of the reactor: e.g.

![Alt Text](https://github.com/MorelliLuca/Neutron-diffusion/blob/master/Output%20examples/filename.gif?raw=true)

The code of `CriticalityFudgeFactor.py` instead allows to evaluate the value of $k_{fudge}$ that makes _critical_ the reactor described in the above `.dat` files.

