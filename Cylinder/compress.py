from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time
import tt

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
r = 1e-3

n_s = n_l
T_s = T_l
   
p_s = gas_params.m * n_s * gas_params.Rg * T_s
    
v_s = np.sqrt(2. * gas_params.Rg * T_s)
mu_s = gas_params.mu(T_s)

l_s = delta * mu_s * v_s / p_s

print 'l_s = ', l_s 

print 'v_s = ', v_s

nv = 44
vmax = 22 * v_s

shape = [2, 2, 11, 2, 2, 11, 2, 2, 11]

hv = 2. * vmax / nv
vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes
    
vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')

flow = tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_l, n_l, u_l, 0., 0., gas_params.Rg))
f_init = lambda x, y, z, vx, vy, vz: flow
f_bound = tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_l, n_l, u_l, 0., 0., gas_params.Rg))
fmax = tt.tensor(Boltzmann_cyl.f_maxwell(vx, vy, vz, T_w, 1., 0., 0., 0., gas_params.Rg))

problem = Boltzmann_cyl.Problem(bc_type_list = ['sym-z', 'in', 'out', 'wall', 'sym-y'],
                                bc_data = [[],
                                           [f_bound],
                                           [f_bound],
                                           [fmax],
                                           []], f_init = f_init)


print 'vmax =', vmax 

CFL = 0.5

mesh = Mesh() 

path = './'
mesh.read_starcd(path, l_s)

L = mesh.nc

S = np.reshape(np.load("file-out-20000.npy"), (L, nv, nv, nv))

S_tt = []

for i in range(L):
    S_tt.append(tt.tensor(S[i, :, :, :]).round(r))

Boltzmann_cyl.save_tt("restart.npy", S_tt, L, nv)
