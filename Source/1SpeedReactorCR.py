import matplotlib.pyplot as plt
import numpy as np
import Reactpy as rt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# ------Code-----

def time_CR(t: float, delta_t: float, t_max :float, k: float)->float:
    if t*delta_t < t_max * 0.1:
        percentage = -10*t/t_max*delta_t +1
    elif t*delta_t < t_max * 0.2:
        percentage = 0
    elif t*delta_t < t_max * 0.3:
        percentage = (t/t_max*delta_t - 0.2)*k*10
    elif t*delta_t < t_max * 0.8:
        percentage = k
    else:
        percentage = (1-k)*5*(t/t_max*delta_t - 0.8) + k
    return percentage

# -----Parameters------
t_max: float = 10
delta_t = 0.1
omega = 1
conv_criterion = 1e-5
D = 1
v = 1
fudge = 1
Delta = 1

input_file = open("Parameters.dat", "r")
for line in input_file:
    line = line.replace("\n", "")
    line = line.replace(" ", "")
    parameter = line.split("=")
    if parameter[0] == "t_max":
        t_max = float(parameter[1])
    elif parameter[0] == "delta_t":
        delta_t = float(parameter[1])
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
    else:
        print(
            "Unknow parameter inserted "
            + parameter[0]
            + ".\n Its value will be ignored: please check for spelling errors."
        )


# Spacial depending paramenters aquired from files
Sigma_absorption = np.diag(rt.file_read_as_vector("Sigma_absorption.dat"))
Sigma_fuel = np.diag(rt.file_read_as_vector("Sigma_fuel.dat"))
Control_rods = np.diag(rt.file_read_as_vector("Control_rods.dat"))

# Creates enviroment for numerical integration
reactor = rt.Grid(grid_matrix=rt.file_read_as_matrix("grid.dat"), Delta=Delta)

# Generates the matrices that represents the discretized differential operators
PDE = (
    -D * (reactor.second_Xderivative_matrix() + reactor.second_Yderivative_matrix())
    + Sigma_absorption
    - v * Sigma_fuel / fudge
)
time_PDE_Matrix = PDE + reactor.flux_PDE_matrix() / (v * delta_t)

data = [reactor.flux_matrix()]  # Set of neutron fluxes at different time

# Initalizes the numerical integrataor
solver = rt.Solver(
    reactor, sources=data[0] / (v * delta_t), PDE_matrix=time_PDE_Matrix + Control_rods
)

# ---Numerical integration---
for t in tqdm(range(int(t_max / delta_t))):
    solver.solve(omega, conv_criterion, update=True)
    data.append(solver.grid.flux_matrix())
    # The solver sources are updated with the new fluxes to obtain time derivative terms
    solver = rt.Solver(
        reactor,
        sources=solver.grid.flux_matrix() / (v * delta_t),
        PDE_matrix=time_PDE_Matrix
        + Control_rods * time_CR(t,delta_t, t_max, 0.2),
    )

# -----Plotting and image genetation-----
data.pop(0)  # Removes the initial condition from the plotte data
max_flux = np.max(data)
min_flux = np.min(data)
fig, ax = plt.subplots()

pcm = ax.pcolormesh(data[0], vmin=min_flux, vmax=max_flux, cmap="turbo", shading="flat")
cbar = plt.colorbar(pcm, ax=ax)


def update(frame):
    pcm = ax.pcolormesh(
        data[frame], vmin=min_flux, vmax=max_flux, cmap="turbo", shading="flat"
    )
    ax.set_title("Time(s):" + str(np.round(frame * delta_t, 3)))
    cbar.update_normal(pcm)
    return [pcm, cbar]


fig.subplots_adjust(top=0.9)

description = (
    r"$D=$"
    + str(D)
    + r"   $v=$"
    + str(v)
    + r"   $k_{fudge}=$"
    + str(fudge)
    + "\n"
    + r"$\Delta=$"
    + str(Delta)
    + r"      $\Delta t=$"
    + str(delta_t)
    + r"$\omega=$"
    + str(omega)
    + r"     conv. crit.$=$"
    + str(conv_criterion)
)
fig.text(
    0.6,
    0.92,
    description,
    fontsize=7,
    bbox=dict(facecolor="white", edgecolor="black", pad=3.0),
)

ani = FuncAnimation(fig, update, frames=len(data), interval=0.3 / delta_t)
# ani.save('filename.gif', writer='imagemagick')

plt.draw()
plt.show()
