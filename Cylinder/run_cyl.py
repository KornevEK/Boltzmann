from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time

from read_starcd import Mesh
from read_starcd import write_tecplot

import functions_cyl as Boltzmann_cyl

# compute parameters for flow around cylinder

# Parameters for argon (default)
gas_params = Boltzmann_cyl.GasParams()

Mach = 10.
Kn = 0.564
delta = 8.0 / (5 * np.pi**0.5 * Kn)
n_l = 2e+23
T_l = 200.
u_l = Mach * ((gas_params.g * gas_params.Rg * T_l) ** 0.5)
T_w = 5.0 * T_l

f_init = lambda x, y, z, vx, vy, vz: Boltzmann_cyl.f_maxwell(vx, vy, vz, T_l, n_l, u_l, 0., 0., gas_params.Rg)

problem = Boltzmann_cyl.Problem(bc_type_list = ['sym-z', 'in', 'out', 'wall', 'sym-y'],
                                bc_data = [[],
                                           [n_l, u_l, 0., 0., T_l],
                                           [n_l, u_l, 0., 0., T_l],
                                           [T_w],
                                           []], f_init = f_init)
n_s = n_l
T_s = T_l
   
p_s = gas_params.m * n_s * gas_params.Rg * T_s
    
v_s = np.sqrt(2. * gas_params.Rg * T_s)
mu_s = gas_params.mu(T_s)

l_s = delta * mu_s * v_s / p_s

print('l_s = ', l_s)

print('v_s = ', v_s) 

nv = 64
vmax = 22 * v_s

print('vmax =', vmax)

CFL = 0.5

mesh = Mesh() 

path = './'
mesh.read_starcd(path, l_s)

L = mesh.nc

print 'Mach =', Mach
nt = 7000

S = Boltzmann_cyl.solver(gas_params, problem, mesh, nt, vmax, nv, CFL, filename = 'file-out.npy', init = 'macroparameters_data_init.txt')

fig, ax = plt.subplots(figsize = (20,10))

colors = cm.rainbow((S.T - np.min(S.T))/(np.max(S.T) - np.min(S.T)))
ax.scatter(mesh.cell_center_coo[:, 0] / l_s, mesh.cell_center_coo[:, 1] / l_s, s = 10, c=colors)
plt.savefig('scatter.png')
plt.close()

fig, ax = plt.subplots(figsize = (20,10))
line, = ax.semilogy(S.frob_norm_iter/S.frob_norm_iter[0])
ax.set(title='$Steps =$' + str(nt))
plt.savefig('norm_iter.png')
plt.close()

#fig, ax = plt.subplots(figsize = (20,10))
#line, = ax.semilogy(mesh.cell_center_coo[:, 0] / l, S.Frob_norm_RHS)
#plt.savefig('norm_rhs.png')

data = np.zeros((L, 7))
    
data[:, 0] = S.n[:]
data[:, 1] = S.ux[:]
data[:, 2] = S.uy[:]
data[:, 3] = S.uz[:]
data[:, 4] = S.p[:]
data[:, 5] = S.T[:]
data[:, 6] = np.zeros(mesh.nc)

write_tecplot(mesh, data, 'cyl.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))
