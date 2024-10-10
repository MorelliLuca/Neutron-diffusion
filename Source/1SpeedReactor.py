import matplotlib.pyplot as plt
import numpy as np
import Reactpy as rt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

#------Code-----

#-----Parameters------ 
t_max: float = 10           
delta_t = 0.1               
omega = 1.1                 
conv_criterion = 10E-5
D = 1
v = 1
fudge = 0.0081056946914


#Spacial depending paramenters aquired from files
Sigma_absorption = np.diag(rt.file_read_as_vector("Sigma_absorption.dat"))
Sigma_fuel = np.diag(rt.file_read_as_vector("Sigma_fuel.dat"))

#Creates enviroment for numerical integration 
reactor = rt.Grid(grid_matrix=rt.file_read_as_matrix("grid.dat"), Delta=0.01)

#Generates the matrices that represents the discretized differential operators
PDE = -D * (reactor.second_Xderivative_matrix() + reactor.second_Yderivative_matrix()) + Sigma_absorption  - v * Sigma_fuel / fudge
time_PDE_Matrix = PDE + reactor.flux_PDE_matrix() / (v * delta_t)

data = [reactor.flux_matrix()]      #Set of neutron fluxes at different time        

#Initalizes the numerical integrataor
solver = rt.Solver(reactor, sources=data[0] / (v * delta_t), PDE_matrix=time_PDE_Matrix)

#Numerical integration
for t in tqdm(range(int(t_max / delta_t))): 
    solver.solve(omega, conv_criterion, update=True)
    data.append(solver.grid.flux_matrix())
    #The solver sources are updated with the new fluxes to obtain time derivative terms
    solver.sources = solver.grid.flux_matrix() / (v * delta_t)

#-----Plotting and image genetation-----
max_flux = np.max(data)
min_flux = np.min(data)
fig, ax = plt.subplots()

data.pop(0) #Removes the initial condition from the plotte data 
pcm = ax.pcolormesh(data[0], vmin = min_flux, vmax = max_flux, cmap='turbo', shading='flat')
cbar = plt.colorbar(pcm, ax=ax)

def update(frame):
   pcm = ax.pcolormesh(data[frame], vmin = min_flux, vmax = max_flux, cmap='turbo', shading='flat')
   ax.set_title('Time(s):' + str(np.round(frame * delta_t, 3)))
   cbar.update_normal(pcm)
   return [pcm, cbar]
ani = FuncAnimation(fig, update, frames=len(data), interval= .3 / delta_t)
ani.save('filename.gif', writer='imagemagick')

plt.draw()
plt.show()
        

    
    



