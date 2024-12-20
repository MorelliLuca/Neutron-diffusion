import matplotlib.pyplot as plt
import numpy as np
import Reactpy as rt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# ------Code-----

# -----Parameters------
omega = 1.1
conv_criterion = 1e-5
D = 1
v = 1
fudge = 1
Delta = 1
intergration_mode = "nopython"
SS_Control_Rods_lvl = 0  # Control rods level at steady state operation

print("Loading parameters from file...")
input_file = open("Parameters.dat", "r")
for line in input_file:
    line = line.replace("\n", "")
    line = line.replace(" ", "")
    parameter = line.split("=")
    if parameter[0] == "t_max":
        continue
    elif parameter[0] == "delta_t":
        continue
    elif parameter[0] == "omega":
        omega = float(parameter[1])
    elif parameter[0] == "conv_criterion":
        conv_criterion = float(parameter[1])
    elif parameter[0] == "D":
        D = float(parameter[1])
    elif parameter[0] == "v":
        v = float(parameter[1])
    elif parameter[0] == "fudge":
        fudge = float(parameter[1])
    elif parameter[0] == "Delta":
        Delta = float(parameter[1])
    elif parameter[0] == "Integration_mode":
        intergration_mode = str(parameter[1])
    elif parameter[0] == "Steady_state_Control_rods_level":
        SS_Control_Rods_lvl = float(parameter[1])
    else:
        print(
            "Unknow parameter inserted "
            + parameter[0]
            + ".\n Its value will be ignored: please check for spelling errors."
        )

# Spacial depending paramenters aquired from files
print("Loading parameters with spacial dependece from files...")
Sigma_absorption = np.diag(rt.file_read_as_vector("Sigma_absorption.dat"))
Sigma_fuel = np.diag(rt.file_read_as_vector("Sigma_fuel.dat"))
Control_rods = np.diag(rt.file_read_as_vector("Control_rods.dat"))

# Creates enviroment for numerical integration
print("Initalizing the integration grid...")
reactor = rt.Grid(rt.file_read_as_matrix("grid.dat"), Delta=Delta)  # change to reactor

# Generates the matrices that represents the discretized differential operators
print("Generating PDE matrix...")
PDE = (
    -D * (reactor.second_Xderivative_matrix() + reactor.second_Yderivative_matrix())
    + Sigma_absorption
    + Control_rods * SS_Control_Rods_lvl
)
# Initalizes the numerical integrataor
solver = rt.Solver(
    reactor,
    rt.vector_to_matrix(
        v / fudge * np.dot(Sigma_fuel, reactor.flux_vector()), reactor.size
    ),
    PDE_matrix=PDE,
)

# ---Numerical integration---
print("---------------------\nINTEGRATION:" + intergration_mode)
not_converged = True
while not_converged:
    new_grid = solver.solve(omega, conv_criterion, False, intergration_mode)
    # Update fudge with stationary solution
    fudge *= np.sum(np.dot(Sigma_fuel, new_grid.flux_vector())) / (
        np.sum(np.dot(Sigma_fuel, solver.grid.flux_vector()))
    )
    if (
        np.linalg.norm(new_grid.flux_vector() - solver.grid.flux_vector())
        < conv_criterion
    ):
        not_converged = False
    # Temporarily disabled for performace
    # PDE = -D * (new_grid.second_Xderivative_matrix() + new_grid.second_Yderivative_matrix()) + Sigma_absorption

    # Update solver with new sources from the stationary solution
    solver = rt.Solver(
        new_grid,
        rt.vector_to_matrix(
            v / fudge * np.dot(Sigma_fuel, new_grid.flux_vector()), new_grid.size
        ),
        PDE_matrix=PDE,
    )

print("---------------------\nThe k_fudge needed to reach criticality is:" + str(fudge))

# -----Plotting and image genetation-----
fig, ax = plt.subplots()

ax.set_title("Stationary critical solution")
pcm = ax.pcolormesh(solver.grid.flux_matrix(), cmap="turbo", shading="flat")
cbar = plt.colorbar(pcm, ax=ax)

fig.subplots_adjust(top=0.9)

description = (
    r"$D=$"
    + str(D)
    + r"   $v=$"
    + str(v)
    + r"   $\omega=$"
    + str(omega)
    + r"   conv. crit.$=$"
    + str(conv_criterion)
    + "\n"
    + r"$\Delta=$"
    + str(Delta)
    + r"   $k_{fudge}=$"
    + str(fudge)
)
fig.text(
    0.62,
    0.94,
    description,
    fontsize=7.5,
    bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
)

plt.draw()
plt.show()
