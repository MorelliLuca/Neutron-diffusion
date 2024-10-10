import matplotlib.pyplot as plt
import numpy as np
import Reactpy as rt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

#------Code-----

omega = 1.1
conv_criterion = 10E-10

Sigma_absorption = np.diag(rt.file_read_as_vector("Sigma_absorption.dat"))
Sigma_fuel = np.diag(rt.file_read_as_vector("Sigma_fuel.dat"))
D = 1
v = 1
fudge = 1

grid_matrix = rt.file_read_as_matrix("grid.dat")

grid = rt.Grid(grid_matrix, 0.01) #change to reactor

PDE = -D * (grid.second_Xderivative_matrix() + grid.second_Yderivative_matrix()) + Sigma_absorption  
solver = rt.Solver(grid, rt.vector_to_matrix(v / fudge * np.dot(Sigma_fuel,grid.flux_vector()), grid.size) ,PDE_matrix=PDE)


while True:
    new_grid = solver.solve(omega, conv_criterion, False)
    fudge *= np.sum(np.dot(Sigma_fuel,new_grid.flux_vector())) / (np.sum(np.dot(Sigma_fuel,solver.grid.flux_vector())))
    if np.linalg.norm(new_grid.flux_vector() - solver.grid.flux_vector()) < conv_criterion:
        break
    PDE = -D * (new_grid.second_Xderivative_matrix() + new_grid.second_Yderivative_matrix()) + Sigma_absorption  
    solver = rt.Solver(new_grid, rt.vector_to_matrix(v / fudge * np.dot(Sigma_fuel,new_grid.flux_vector()), grid.size) ,PDE_matrix=PDE)

print(fudge)


fig, ax = plt.subplots()

ax.set_title('Stationary critical solution')
pcm = ax.pcolormesh(solver.grid.flux_matrix(), cmap='turbo', shading='flat')
cbar = plt.colorbar(pcm, ax=ax)

plt.draw()
plt.show()
        

    
    



