# How to use
This directory contains 3 codes that achieve different tasks:
- `CriticalityFudgeFactor.py`: computes the fudge factor of a given reactor configuration,
- `1SpeedReactor.py`: integrates over time the neutron flux dynamics of a given 1 speed reactor configuration,
- `1SpeedReactorCR.py`: as the above but also _control rods_ dynamics is included in the integration.

All the parameters for these codes must be inserted in apposite `.dat` files in this directory:
- `Parameters.dat`: contains all the simulation parameters and options,
- `grid.dat`: contains the geometrical information of the reactor configuration space and the initial conditions,
- `Sigma_absorption.dat`: contains the absorption values of the moderator over the configuration space,
- `Sigma_fuel.dat`: contains the production values of the fuel over the configuration space,
- `Control_rods.dat`: contains the maximum absorption values of the control rods over the configuration space.

All these files are shared between all the codes.
## Parameters insertion
`Parameters.dat` contains the following variables:
- *t_max* is the time duration (*s*) of time integration,
- *delta_t* is the time discretization step,
- *omega* is the weight used in _successive relaxation method_
- *D* is the diffusion parameter,
- *v* is the average speed of neutron,
- *fudge* is the fudge factor
- *Delta* is the spacial discretization step.

        t_max = 10         
        delta_t = 0.1               
        omega = 1               
        conv_criterion = 1E-5
        D = 10
        v = 1
        fudge = 0.045210616027732034
        Delta = 0.2
All these parameters can be inserted in any order and spaces can be omitted. Any unknown parameter will be ignored by the code and for every missing parameter the code will use default values.

`grid.dat` must be set in a matrix form, in the following way:

    E E E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E E E
    E E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E E
    E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E
    E E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E E
    E E E 1 1 1 1 1 1 1 1 1 1 1 1 1 1 E E E
Each element of the matrix must be separated by a `space` char. Numerical entries will be interpreted as initial condition (or guess) of the neutron flux, the char `E` is instead interpreted as an empty cell which flux is always zero (Boundary condition). Any other value is rejected as invalid.

`Sigma_absorption.dat`, `Sigma_fuel.dat` and `Control_rods.dat` must be set as the above but only numerical values are allowed (set zero values for the corresponding empty cells).

