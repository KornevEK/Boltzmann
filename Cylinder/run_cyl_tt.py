from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time

from read_starcd import Mesh
from read_starcd import write_tecplot

import functions_cyl_tt as Boltzmann_cyl_tt

class Params(object):
 
    def __init__(self):
        #fundamental constants
        self.Na = 6.02214129e+23
        self.kB = 1.381e-23 # J / K
        #gas parameters
        self.Mol = 40e-3 # kg / mol
        self.Rg = 8.3144598  / self.Mol  # J / (kg * K) 
        self.m = self.Mol / self.Na # kg
    
        self.Pr = 2. / 3.
        self.C = 144.4
        self.T_0 = 273.11
        self.mu_0 = 2.125e-05
    
        self.g = 5. / 3.
        
        self.d = 3418e-13

p = Params()



Tau = 10000

M = 10.
Kn = 0.564


n_l = 2e+23 
T_l = 300.
T_wall = T_l*5

n_s = n_l
T_s = T_l
l_s = 1.#100*l
   
p_s = p.m * n_s * p.Rg * T_s
    
v_s = np.sqrt(2. * p.Rg * T_s)

#l = 1. / ((2 ** .5) * np.pi * n_l * p.d * p.d)
  
lambda_s = Kn * l_s


N = 45
vmax = 22 * v_s


CFL = 0.5

r = 1e-7

mesh = Mesh() 

path = './'
mesh.read_starcd(path, l_s)

L = mesh.nc

print 'Max =', M

S = Boltzmann_cyl_tt.solver_tt(p=p, mesh=mesh, M=M, Kn=Kn, n_l=n_l, T_l=T_l, T_wall=T_wall, Tau=Tau, vmax=vmax, N=N,
           CFL=CFL, r, filename = 'file.txt')


           
           
fig, ax = plt.subplots(figsize = (20,10))

colors = cm.rainbow((S.Temp - np.min(S.Temp))/(np.max(S.Temp) - np.min(S.Temp)))
ax.scatter(mesh.cell_center_coo[:, 0] / l_s, mesh.cell_center_coo[:, 1] / l_s, s = 10, c=colors)
plt.savefig('scatter.png')

fig, ax = plt.subplots(figsize = (20,10))
line, = ax.semilogy(S.Frob_norm_iter)
ax.set(title='$Steps =$' + str(Tau))
plt.savefig('norm_iter.png')

#fig, ax = plt.subplots(figsize = (20,10))
#line, = ax.semilogy(mesh.cell_center_coo[:, 0] / l, S.Frob_norm_RHS)
#plt.savefig('norm_rhs.png')

data = np.zeros((L, 3))
    
data[:, 0] = S.Dens[:]
data[:, 1] = S.Vel[:]
data[:, 2] = S.Temp[:]

write_tecplot(mesh, data, 'cyl.dat', ('Dens', 'Vel', 'Temp'))
