from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import tt
import time
from read_starcd import write_tecplot

def f_maxwell(vx, vy, vz, T, n, ux, uy, uz, Rg):
    """Compute maxwell distribution function on cartesian velocity mesh
    
    vx, vy, vz - 3d numpy arrays with x, y, z components of velocity mesh
    in each node
    T - float, temperature in K
    n - float, numerical density
    ux, uy, uz - floats, x,y,z components of equilibrium velocity
    Rg - gas constant for specific gas
    """
    return n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * (np.exp(-((vx - ux)**2 + (vy - uy)**2 + (vz - uz)**2) / (2. * Rg * T)))

class GasParams:
    Na = 6.02214129e+23 # Avogadro constant
    kB = 1.381e-23 # Boltzmann constant, J / K
    Ru = 8.3144598 # Universal gas constant

    def __init__(self, Mol = 40e-3, Pr = 2. / 3., g = 5. / 3., d = 3418e-13):
        self.Mol = Mol
        self.Rg = self.Ru  / self.Mol  # J / (kg * K) 
        self.m = self.Mol / self.Na # kg
    
        self.Pr = Pr
        
        self.C = 144.4
        self.T_0 = 273.11
        self.mu_0 = 2.125e-05
        self.mu_suth = lambda T: self.mu_0 * ((self.T_0 + self.C) / (T + self.C)) * ((T / self.T_0) ** (3. / 2.))
        self.mu = lambda T: self.mu_suth(200.) * (T/200.)**0.734
        self.g = g # specific heat ratio
        self.d = d # diameter of molecule
        
class Problem:
    def __init__(self, bc_type_list = None, bc_data = None, f_init = None):
        # list of boundary conditions' types
        # acording to order in starcd '.bnd' file
        # list of strings
        self.bc_type_list = bc_type_list
        # data for b.c.: wall temperature, inlet n, u, T and so on.
        # list of lists
        self.bc_data = bc_data
        # Function to set initial condition
        self.f_init = f_init
        
def set_bc(gas_params, bc_type, bc_data, f, vx, vy, vz, v_nil, v_nil_plus, v_nil_minus, r):
    """Set boundary condition
    """
    if (bc_type == 'sym-x'): # symmetry in x
        l = f.to_list(f)
        l[0] = l[0][:,::-1,:]
        f = f.from_list(l)
        return f
    elif (bc_type == 'sym-y'): # symmetry in y
        l = f.to_list(f)
        l[1] = l[1][:,::-1,:]
        f = f.from_list(l)
        return f
    elif (bc_type == 'sym-z'): # symmetry in z
        l = f.to_list(f)
        l[2] = l[2][:,::-1,:]
        f = f.from_list(l)
        return f
    elif (bc_type == 'in'): # inlet
        # unpack bc_data
        f_bound =  bc_data[0]
        return f_bound
    elif (bc_type == 'out'): # outlet
        # unpack bc_data
        f_bound =  bc_data[0]
        return f_bound
    elif (bc_type == 'wall'): # wall
        # unpack bc_data
        fmax = bc_data[0]
        hv = vx[1, 0, 0] - vx[0, 0, 0]
        Ni = (hv**3) * tt.sum((f * v_nil_plus).round(r))
        Nr = (hv**3) * tt.sum((fmax * v_nil_minus).round(r))
        # TODO: replace np.sqrt(2 * np.pi / (gas_params.Rg * T_w))
        # with discrete quarature, as in the dissertation
        n_wall = - Ni/ Nr
#        n_wall = 2e+23 # temprorary
        return n_wall * fmax
            
def comp_macro_param_and_j(f, vx_, vx, vy, vz, vx_tt, vy_tt, vz_tt, v2, gas_params, r, F, one, ranks):
    # takes "F" with (1, 1, 1, 1) ranks to perform to_list function
    # takes precomputed "one" tensor
    # "ranks" array for mean ranks
    # much less rounding
    Rg = gas_params.Rg
    hv = vx_[1] - vx_[0]
    n = (hv ** 3) * tt.sum(f)

    ux = (1. / n) * (hv ** 3) * tt.sum(vx_tt * f)
    uy = (1. / n) * (hv ** 3) * tt.sum(vy_tt * f)
    uz = (1. / n) * (hv ** 3) * tt.sum(vz_tt * f)
    
    u2 = ux*ux + uy*uy + uz*uz
    
    T = (1. / (3. * n * Rg)) * ((hv ** 3) * tt.sum((v2 * f)) - n * u2)

    Vx = vx - ux
    Vy = vy - uy
    Vz = vz - uz

    Vx_tt = (vx_tt - ux * one).round(r)
    Vy_tt = (vy_tt - uy * one).round(r)
    Vz_tt = (vz_tt - uz * one).round(r)

    rho = gas_params.m * n

    p = rho * Rg * T

    cx = Vx_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    cy = Vy_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    cz = Vz_tt * (1. / ((2. * Rg * T) ** (1. / 2.)))
    # TODO: do rounding after each operation
    c2 = ((cx*cx) + (cy*cy) + (cz*cz)).round(r)
    
    Sx = (1. / n) * (hv ** 3) * tt.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * tt.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * tt.sum(cz * c2 * f)

    mu = gas_params.mu(T)

    fmax = F
    fmax_list = F.to_list(F)
    fmax_list[0][0, :, 0] = np.exp(-((vx_ - ux) ** 2) / (2. * Rg * T)) # vx_ instead of Vx[0, :, :]
    fmax_list[1][0, :, 0] = np.exp(-((vx_ - uy) ** 2) / (2. * Rg * T))
    fmax_list[2][0, :, 0] = np.exp(-((vx_ - uz) ** 2) / (2. * Rg * T))
    fmax = fmax.from_list(fmax_list)
    fmax = n * ((1. / (2. * np.pi * Rg * T)) ** (3. / 2.)) * fmax
    fmax = fmax.round(r)
	
	# TODO: replace with tt.from_list
    #fmax = tt.tensor(f_maxwell(vx, vy, vz, T, n, ux, uy, uz, gas_params.Rg))
	# TODO: do rounding after each operation
    f_plus = fmax * (one + ((4. / 5.) * (1. - gas_params.Pr) * (cx*Sx + cy*Sy + cz*Sz) * ((c2 - (5. / 2.) * one))))
    ranks[0] += f_plus.erank
    f_plus = f_plus.round(r)
    ranks[1] += f_plus.erank

    J = (f_plus - f) * (p / mu)
    J = J.round(r)
    
    nu = p / mu
    
    return J, n, ux, uy, uz, T, nu, rho, p, ranks

def save_tt(filename, f, L, N):
    
    m = max(f[i].core.size for i in range(L))

    F = np.zeros((m+4, L))
    
    for i in range(L):
        F[:4, i] = f[i].r.ravel()
        F[4:f[i].core.size+4, i] = f[i].core.ravel()
    
    np.save(filename, F)#, fmt='%s')
    
def load_tt(filename, L, N):
    
    F = np.load(filename)
    
    f = list()
    
    for i in range(L):
        
        f.append(tt.rand([N, N, N], 3, F[:4, i]))
        f[i].core = F[4:f[i].core.size+4, i]
        
    return f

def solver(gas_params, problem, mesh, nt, vmax, nv, CFL, r, filename, init = '0'):
    """Solve Boltzmann equation with model collision integral 
    
    gas_params -- object of class GasParams, contains gas parameters and viscosity law
    
    problem -- object of class Problem, contains list of boundary conditions,
    data for b.c., and function for initial condition
    
    mesh - object of class Mesh
    
    nt -- number of time steps
    
    vmax -- maximum velocity in each direction in velocity mesh
    
    nv -- number of nodes in velocity mesh
    
    CFL -- courant number
    
    filename -- name of output file for f
    
    init - name of restart file
    """
    t = time.time()

    log = open('log.txt', 'w') # log file
#TODO
    
#    CPU_time = namedtuple('CPU_time', ['reconstruction', 'boundary', 'fluxes', 'rhs', 'update'])
    
#    CPU = CPU_time(0., 0., 0., 0., 0.)

    cpu_time = np.zeros(10) # sum time for each phase
    cpu_time_name = ['reconstruction', 'boundary', 'fluxes', 'rhs', 'update', 'rhs_j']

    ranks = np.zeros(10)

    h = np.min(mesh.cell_diam)
    tau = h * CFL / (vmax * (3.**0.5))
    
    hv = 2. * vmax / nv
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, nv) # coordinates of velocity nodes
    
    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')

    v_nil = []
#    v_nil_max = np.zeros(mesh.nf)
    v_nil_tmp = np.zeros((nv, nv, nv))
#    v_est_tmp = 0. * tt.ones((nv,nv,nv))
    v_nil_minus = []
    v_nil_plus = []
    v_est = []

    for jf in range(mesh.nf):
        v_nil_tmp = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz
#        v_nil_max[jf] = np.max(np.abs(v_nil_tmp))
        v_nil.append(tt.tensor(v_nil_tmp).round(r))
        v_nil_plus.append(tt.tensor(np.where(v_nil_tmp > 0, v_nil_tmp, 0.)))
        v_nil_minus.append(tt.tensor(np.where(v_nil_tmp < 0, v_nil_tmp, 0.)))
#        v_est_tmp = tt.tensor(np.abs(v_nil_tmp))
        v_est.append(tt.tensor(np.abs(v_nil_tmp)).round(1e-1, rmax = 2)) # rounded abs tensor, 2 arguments because of the bug

    vx_tt = tt.tensor(vx).round(r)
    vy_tt = tt.tensor(vy).round(r)
    vz_tt = tt.tensor(vz).round(r)

    v2 = (vx_tt*vx_tt + vy_tt*vy_tt + vz_tt*vz_tt).round(r)
#    v_est = tt.tensor((vx**2 + vy**2 + vz**2)**0.5, eps = 1e-3)
    v_est_minus = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nf))
    v_est_plus = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nf))

    for jf in range(mesh.nf):
        v_est_minus[jf] = (v_nil[jf] - v_est[jf]).round(r)
        v_est_plus[jf] = (v_nil[jf] + v_est[jf]).round(r)

    v_est_minus_sum = 0.
    v_est_plus_sum = 0.
    for jf in range(mesh.nf):
        v_est_minus_sum += v_est_minus[jf].erank
        v_est_plus_sum += v_est_plus[jf].erank

    log.write("v_est_minus mean rank " + str(v_est_minus_sum / mesh.nf) + "\n")#print "v_est_minus mean rank", v_est_minus_sum / mesh.nf
    log.write("v_est_plus mean rank " + str(v_est_plus_sum / mesh.nf) + "\n")#print "v_est_plus mean rank", v_est_plus_sum / mesh.nf
    
    # set initial condition 
    f = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nc))
    if (init == '0'):
        for i in range(mesh.nc):
            x = mesh.cell_center_coo[i, 0]
            y = mesh.cell_center_coo[i, 1]
            z = mesh.cell_center_coo[i, 2]
            f[i] = problem.f_init(x, y, z, vx, vy, vz)
    else:
#        restart from distribution function
#        f = load_tt(init, mesh.nc, nv)
#        restart form macroparameters array
        init_data = np.loadtxt(init)
        for ic in range(mesh.nc):
            f[ic] = tt.tensor(f_maxwell(vx, vy, vz, init_data[ic, 5], init_data[ic, 0], init_data[ic, 1], init_data[ic, 2], init_data[ic, 3], gas_params.Rg)).round(r)
        
    tmp = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nc))
    # TODO: may be join f_plus and f_minus in one array
    f_plus = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nf)) # Reconstructed values on the right
    f_minus = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nf)) # reconstructed values on the left
    flux = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nf)) # Flux values
    rhs = list(0. * tt.ones((nv,nv,nv)) for i in range(mesh.nc))
    
    # Arrays for macroparameters
    n = np.zeros(mesh.nc)
    rho = np.zeros(mesh.nc)
    ux = np.zeros(mesh.nc)
    uy = np.zeros(mesh.nc)
    uz = np.zeros(mesh.nc)
    p = np.zeros(mesh.nc)
    T = np.zeros(mesh.nc)
    nu = np.zeros(mesh.nc)
    rank = np.zeros(mesh.nc)
    data = np.zeros((mesh.nc, 7))

    F = tt.tensor(f_maxwell(vx, vy, vz, 200., 2e+23, 100., 0., 0., gas_params.Rg)).round(r)
    zero_tensor = 0. * tt.ones((nv,nv,nv))
    one = tt.ones((nv,nv,nv))
    
    frob_norm_rhs = np.zeros(mesh.nc)
    frob_norm_iter = np.array([])

    t_ = time.time() - t
#    print "initialization", t_
    log.write("initialization " + str(t_) + "\n")

    t1 = time.time()
    it = 0
    while(it < nt):
        it += 1
        # reconstruction for inner faces
        # 1st order
        t = time.time()
        for ic in range(mesh.nc):
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                # TODO: think how do this without 'if'
                if (mesh.cell_face_normal_direction[ic, j] == 1):
                    f_minus[jf] = f[ic].copy()
                else:
                    f_plus[jf] = f[ic].copy()
        t_ = time.time() - t
        cpu_time[0] += t_
#        print "reconstruction", it, t_
        log.write("reconstruction " + str(it) + " " + str(t_) + "\n")
                      
        # boundary condition
        # loop over all boundary faces
        t = time.time()
        for j in range(mesh.nbf):
            jf = mesh.bound_face_info[j, 0] # global face index
            bc_num = mesh.bound_face_info[j, 1]
            bc_type = problem.bc_type_list[bc_num]
            bc_data = problem.bc_data[bc_num]
            if (mesh.bound_face_info[j, 2] == 1):
                # TODO: normal velocities v_nil can be pretime = 0 h 5 m 47 s-computed one time
                # then we can pass to function p.bc only v_nil
                f_plus[jf] =  set_bc(gas_params, bc_type, bc_data, f_minus[jf], vx, vy, vz, v_nil[jf], v_nil_plus[jf], v_nil_minus[jf], r)
            else:
                f_minus[jf] = set_bc(gas_params, bc_type, bc_data, f_plus[jf], vx, vy, vz, -v_nil[jf], -v_nil_minus[jf], -v_nil_plus[jf], r)
        t_ = time.time() - t
        cpu_time[1] += t_
#        print "boundary", it, t_
        log.write("boundary " + str(it) + " " + str(t_) + "\n")

        # riemann solver - compute fluxes
        t = time.time()
        for jf in range(mesh.nf):
#            flux[jf] = (1. / 2.) * mesh.face_areas[jf] * (v_nil[jf] * (f_plus[jf] + f_minus[jf]) - v_nil_max[jf] * (f_plus[jf] - f_minus[jf]))
#            Exact flux 
#            flux[jf] = mesh.face_areas[jf] * (v_nil_plus[jf] * f_minus[jf] + v_nil_minus[jf] * f_plus[jf]).round(r)
#            Approximate flux with speed estimate
#            flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((v_nil[jf] * (f_plus[jf] + f_minus[jf]).round(r)).round(r) - (v_est * (f_plus[jf] - f_minus[jf]).round(r)).round(r)).round(r)
#      	     flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((v_nil[jf] * (f_plus[jf] + f_minus[jf])) - (v_est * (f_plus[jf] - f_minus[jf]))).round(r)
	    flux[jf] = (1. / 2.) * mesh.face_areas[jf] * ((f_plus[jf] * v_est_minus[jf]) + (f_minus[jf] * v_est_plus[jf]))
            flux[jf] = flux[jf].round(r)
# TODO: move round to separate string, measure time, store ranks before rounding

        t_ = time.time() - t
        cpu_time[2] += t_
#        print "fluxes", it, t_ 
        log.write("fluxes " + str(it) + " " + str(t_) + "\n")

        # computation of the right-hand side
        t = time.time()
        for ic in range(mesh.nc):
            rhs[ic] = zero_tensor.copy()
            # sum up fluxes from all faces of this cell
            for j in range(6):
				# TODO: do rounding after each binary operation
                jf = mesh.cell_face_list[ic, j]
                rhs[ic] += - (mesh.cell_face_normal_direction[ic, j]) * (1. / mesh.cell_volumes[ic]) * flux[jf]
                rhs[ic] = rhs[ic].round(r)
            # Compute macroparameters and collision integral
            jt = time.time()
            J, n[ic], ux[ic], uy[ic], uz[ic], T[ic], nu[ic], rho[ic], p[ic], ranks = comp_macro_param_and_j(f[ic], vx_, vx, vy, vz, vx_tt, vy_tt, vz_tt, v2, gas_params, r, F, one, ranks)
            rhs[ic] += J
            ranks[2] += rhs[ic].erank
            rhs[ic] = rhs[ic].round(r)
            ranks[3] += rhs[ic].erank
            cpu_time[5] += (time.time() - jt)
        t_ = time.time() - t
        cpu_time[3] += t_
#        print "rhs", it, t_
        log.write("rhs " + str(it) + " " + str(t_) + "\n")
        
        frob_norm_iter = np.append(frob_norm_iter, np.sqrt(sum([(rhs[ic].norm())**2 for ic in range(mesh.nc)])))

        # update values
        t = time.time()
        for ic in range(mesh.nc):
            tmp[ic] = (f[ic] + tau * rhs[ic]).round(r)
            f[ic] = tmp[ic].copy()
            rank[ic] = f[ic].erank
        t_ = time.time() - t
        cpu_time[4] += t_
#        print "update", it, t_
        log.write("update " + str(it) + " " + str(t_) + "\n")
        
        if ((it % 100) == 0):     
            fig, ax = plt.subplots(figsize = (20,10))
            line, = ax.semilogy(frob_norm_iter/frob_norm_iter[0])
            ax.set(title='$Steps =$' + str(it))
            plt.savefig('norm_iter.png')
            plt.close()
                
            fig, ax = plt.subplots(figsize = (20,10))

            colors = cm.rainbow((T - np.min(T))/(np.max(T) - np.min(T)))
            ax.scatter(mesh.cell_center_coo[:, 0], mesh.cell_center_coo[:, 1], s = 10, c=colors)
            ax.set_xlim(np.min(mesh.cell_center_coo[:, 0]), np.max(mesh.cell_center_coo[:, 0]))
            ax.set_ylim(np.min(mesh.cell_center_coo[:, 1]), np.max(mesh.cell_center_coo[:, 1]))
            plt.savefig('scatter_temp_iter=' + str(it) + '.png')
            plt.close()
                            
            data[:, 0] = n[:] # now n instead of rho
            data[:, 1] = ux[:]
            data[:, 2] = uy[:]
            data[:, 3] = uz[:]
            data[:, 4] = p[:]
            data[:, 5] = T[:]
            data[:, 6] = rank[:]
            
            write_tecplot(mesh, data, 'cyl.dat', ('n', 'ux', 'uy', 'uz', 'p', 'T', 'rank'))
        
        
    t2 = time.time() - t1

    t2 = int(round(t2))
    
#    print "time =", t2 / 3600, "h", (t2 % 3600) / 60, "m", t2 % 60, "s"
    log.write("time = " + str(t2 / 3600) + " h " + str((t2 % 3600) / 60) + " m " + str(t2 % 60) + " s " + "\n")
        
    for ic in range(mesh.nc):
        frob_norm_rhs[ic] = rhs[ic].norm()

#    l = 1. / ((2 ** .5) * np.pi * n_l * p.d * p.d)
#    delta = l / (n_r - n_l) * np.max(Dens[1:] - Dens[:-1]) / (2 * h)

    save_tt(filename, f, mesh.nc, nv)
    
    Return = namedtuple('Return', ['f', 'n', 'ux', 'uy', 'uz', 'T', 'p', 'rank', 'frob_norm_iter', 'frob_norm_rhs'])
    
    S = Return(f, n, ux, uy, uz, T, p, rank, frob_norm_iter, frob_norm_rhs)

    for i in range(len(cpu_time_name)):
#        print cpu_time[i], cpu_time_name[i]
        log.write(str(cpu_time[i]) + " " + cpu_time_name[i] + "\n")

    print ranks / nt / mesh.nc

    log.close()
    
    return S
