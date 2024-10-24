"""
This package provides the tools the are needed in order to simulate numerically a nucler fission reactor.

Modules:
--------
- `ControlRods`: provides functions that can generate predefined controld rods dynamics.
- `Functions`: useful fucntions.
- `IntegrationsMethods`: different optimizations for linear algebra methods used during integration.
- `SolvePDE`: all you need to numerically integrate the PDE of the neutron diffusion of a nucleat reactor.

"""

from .SolvePDE import *
from .Functions import *
