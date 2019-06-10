from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
import time

def rs(f1, f2, f, vx_, N):
    for j in range(N):
        if (vx_[j] > 0.):
            f[j, :, :] = vx_[j] * f1[j, :, :]
        else:
            f[j, :, :] = vx_[j] * f2[j, :, :]
    return f
    
def F_m(vx, vy, vz, T, n, p):
    return n * ((1. / (2. * np.pi * p.Rg * T)) ** (3. / 2.)) * (np.exp(-(vx*vx + vy*vy + vz*vz) / (2. * p.Rg * T)))

def J_tt(f, vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p):
    
    n = (hv ** 3) * tt.sum(f)

    ux = (1. / n) * (hv ** 3) * tt.sum(vx_tt * f)
    uy = (1. / n) * (hv ** 3) * tt.sum(vy_tt * f)
    uz = (1. / n) * (hv ** 3) * tt.sum(vz_tt * f)
    
    v2 = (vx_tt*vx_tt + vy_tt*vy_tt + vz_tt*vz_tt).round(r)
    u2 = ux*ux + uy*uy + uz*uz
    
    T = (1. / (3. * n * p.Rg)) * ((hv ** 3) * tt.sum(v2 * f) - n * u2)

    Vx = vx - ux
    Vy = vy - uy
    Vz = vz - uz
    
    Vx_tt = (vx_tt - ux * tt.ones((N,N,N))).round(r)
    Vy_tt = (vy_tt - uy * tt.ones((N,N,N))).round(r)
    Vz_tt = (vz_tt - uz * tt.ones((N,N,N))).round(r)

    rho = p.m * n

    P = rho * p.Rg * T

    cx = Vx_tt * (1. / ((2. * p.Rg * T) ** (1. / 2.)))
    cy = Vy_tt * (1. / ((2. * p.Rg * T) ** (1. / 2.)))   
    cz = Vz_tt * (1. / ((2. * p.Rg * T) ** (1. / 2.)))
    
    c2 = (cx*cx + cy*cy + cz*cz).round(r)

    Sx = (1. / n) * (hv ** 3) * tt.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * tt.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * tt.sum(cz * c2 * f)

    mu = p.mu_0 * ((p.T_0 + p.C) / (T + p.C)) * ((T / p.T_0) ** (3. / 2.))

    F_M = tt.tensor(F_m(Vx, Vy, Vz, T, n, p))
    
    f_plus = F_M * (tt.ones((N,N,N)) + (4. / 5.) * (1. - p.Pr) * (cx*Sx + cy*Sy + cz*Sz) * (c2 - (5. / 2.) * tt.ones((N,N,N))))
    f_plus = f_plus.round(r)
    
    J = (f_plus - f) * (P / mu)
    J = J.round(r)
    
    nu = P / mu
    
    return J, n, ux, T, nu

def save_tt(filename, f, L, N):
    
    m = max(f[i].core.size for i in range(L))

    F = np.zeros((m+4, L))
    
    for i in range(L):
        F[:4, i] = f[i].r.ravel()
        F[4:f[i].core.size+4, i] = f[i].core.ravel()
    
    np.savetxt(filename, F)#, fmt='%s')
    
def load_tt(filename, L, N):
    
    F = np.loadtxt(filename)
    
    f = list()
    
    for i in range(L):
        
        f.append(tt.rand([N, N, N], 3, F[:4, i]))
        f[i].core = F[4:f[i].core.size+4, i]
        
    return f

def solver_tt(p, mesh, M, Kn, n_l, T_l, T_wall, Tau, vmax, N, CFL, r, filename, init = '0'):
    
    n_s = n_l
    T_s = T_l
    l_s = p.d
    
    p_s = p.m * n_s * p.Rg * T_s
    
    v_s = np.sqrt(2. * p.Rg * T_s)
    
    lambda_s = Kn * l_s
    
    u_l = M * ((p.g * p.Rg * T_l) ** .5)
    
    # TODO add calculation of cell diameter
    h = np.min(mesh.cell_diam)
    tau = h * CFL / vmax
    
    t = 0.
    
    hv = 2. * vmax / N
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, N)
    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')
    L = mesh.nc
    
    v_nil = []
    v_nil_max = np.zeros(mesh.nf)
    v_nil_tmp = np.zeros((N, N, N))
    v_nil_minus = []
    v_nil_plus = []
    
    for jf in range(mesh.nf):
        v_nil_tmp = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz
        v_nil_max[jf] = np.max(np.abs(v_nil_tmp))
        v_nil.append(tt.tensor(v_nil_tmp))
        v_nil_plus.append(tt.tensor(np.where(v_nil_tmp > 0, v_nil, 0.)))
        v_nil_minus.append(tt.tensor(np.where(v_nil_tmp < 0, -v_nil, 0.)))
    
        
    vx_tt = tt.tensor(vx)
    vy_tt = tt.tensor(vy)
    vz_tt = tt.tensor(vz)
    
    F_l = tt.tensor(F_m(vx-u_l, vy, vz, T_l, n_l, p))
    
    F_wall = tt.tensor(F_m(vx, vy, vz, T_wall, 1., p))
    
    # initial condition 
    f = list(0. * tt.ones((N,N,N)) for i in range(L))
    
    if (init == '0'):
        for i in range(L):
            f[i] = F_l
            
    else:
        f = load_tt(init, L, N)
    
    
#    for i in range(L):
#        f[i, :, :, :] = problem.set_init_cond(mesh.cell_center_coo[i])
    
    tmp = list(0. * tt.ones((N,N,N)) for i in range(L))
    # TODO: may be join f_plus and f_minus in one array
    f_plus = list(0. * tt.ones((N,N,N)) for i in range(mesh.nf))
    f_minus = list(0. * tt.ones((N,N,N)) for i in range(mesh.nf))
    Flow = list(0. * tt.ones((N,N,N)) for i in range(mesh.nf))
    RHS = list(0. * tt.ones((N,N,N)) for i in range(L))
    
    j = list(0. for i in range(L))
    jj = list(0. * tt.ones((N,N,N)) for i in range(L))
    
    Dens = np.zeros(L)
    Vel = np.zeros(L)
    Temp = np.zeros(L)
    
    Frob_norm_RHS = np.zeros(L)
    Frob_norm_iter = np.array([])
    
    t1 = time.clock()
    
    while(t < Tau*tau):
        t += tau
        # reconstruction for inner faces
        # 1st order
        for ic in range(L):
            for j in range(6):
                jf = mesh.cell_face_list[ic, j]
                # TODO: think how do this without 'if'
                if (mesh.cell_face_normal_direction[ic, j] == 1):
                    f_minus[jf] = f[ic]
                else:
                    f_plus[jf] = f[ic]
                
        # boundary condition
        # loop over all boundary faces
        for j in range(mesh.nbf):
            jf = mesh.bound_face_info[j, 0] # global face index
                # TODO: think how do this without 'if'
            if (mesh.bound_face_info[j, 1] == 0): # symmetry
                if (mesh.bound_face_info[j, 2] == 1):
                    f_plus[jf] = f_minus[jf]
                else:
                    f_minus[jf] = f_plus[jf]
            elif (mesh.bound_face_info[j, 1] == 1): # inlet
                if (mesh.bound_face_info[j, 2] == 1):
                    f_plus[jf] = F_l
                else:
                    f_minus[jf] = F_l
            elif (mesh.bound_face_info[j, 1] == 2): # outlet
                if (mesh.bound_face_info[j, 2] == 1):
                    f_plus[jf] = F_l
                else:
                    f_minus[jf] = F_l
            elif (mesh.bound_face_info[j, 1] == 3): # wall
                if (mesh.bound_face_info[j, 2] == 1):
                    #n_wall = J(f_minus[jf, :, :, :], vx, vy, vz, hv, N, p)[1]
                    Ni = hv**3 * tt.sum(f_minus[jf] * v_nil_plus[jf])
                    n_wall = np.sqrt(2 * np.pi/ (p.Rg * T_wall)) * Ni
                    f_plus[jf] = n_wall * F_wall
                else:
                    #n_wall = J(f_plus[jf, :, :, :], vx, vy, vz, hv, N, p)[1]
                    Ni = hv**3 * tt.sum(f_plus[jf] * v_nil_minus[jf])
                    n_wall = np.sqrt(2 * np.pi/ (p.Rg * T_wall)) * Ni
                    f_minus[jf] = n_wall * F_wall
                

        
        # riemann solver - compute fluxes
        for jf in range(mesh.nf):
#            v_nil = mesh.face_normals[jf, 0] * vx + mesh.face_normals[jf, 1] * vy + mesh.face_normals[jf, 2] * vz
#            v_nil_max = np.max(np.abs(v_nil))
            Flow[jf] = (1. / 2.) * mesh.face_areas[jf] * (v_nil[jf] * (f_plus[jf] + f_minus[jf]) - v_nil_max[jf] * (f_plus[jf] - f_minus[jf]))
#            Flow[jf] = mesh.face_areas[jf] * (v_nil_plus[jf] * f_minus[jf] + v_nil_minus[jf] * f_plus[jf])
            Flow[jf] = Flow[jf].round(r)
                
        
        for ic in range(L):
            RHS[ic] *= 0.
            for lf in range(6):
                jf = mesh.cell_face_list[ic, lf]
                RHS[ic] += - (mesh.cell_face_normal_direction[ic, lf]) * (1. / mesh.cell_volumes[ic]) * Flow[jf]
#            j[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)
#            jj[ic] = j[ic][0]
#            jj[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[0]
            RHS[ic] += J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[0]#jj[ic]
            RHS[ic] = RHS[ic].round(r)
        
            
        Frob_norm_iter = np.append(Frob_norm_iter, sum([RHS[ic].norm() for ic in range(L)]))


        # update values
        for ic in range(L):
            tmp[ic] = f[ic] + tau * RHS[ic]
            tmp[ic] = tmp[ic].round(r)
                
        f = tmp
        
        if ((int(t/tau) % 100) == 0):     
            fig, ax = plt.subplots(figsize = (20,10))
            line, = ax.semilogy(Frob_norm_iter)
            ax.set(title='$Steps =$' + str(int(t/tau)))
            plt.savefig('norm_iter.png')
            plt.close()

            for ic in range(L):
                Dens[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[1]
                Vel[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[2]
                Temp[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[3]
            
            fig, ax = plt.subplots(figsize = (20,10))

            colors = cm.rainbow((Temp - np.min(Temp))/(np.max(Temp) - np.min(Temp)))
            ax.scatter(mesh.cell_center_coo[:, 0] / l_s, mesh.cell_center_coo[:, 1] / l_s, s = 10, c=colors)
            plt.savefig('scatter_temp_iter=' + str(t//tau) + '.png')
            plt.close()
        
        
    t2 = time.clock() - t1

    t2 = int(round(t2))
    
    print "time =", t2 / 3600, "h", (t2 % 3600) / 60, "m", t2 % 60, "s"

    for ic in range(L):
        Dens[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[1]
        Vel[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[2]
        Temp[ic] = J_tt(f[ic], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)[3]
        
    for ic in range(L):
        Frob_norm_RHS[ic] = RHS[ic].norm()

#    l = 1. / ((2 ** .5) * np.pi * n_l * p.d * p.d)
        
#    delta = l / (n_r - n_l) * np.max(Dens[1:] - Dens[:-1]) / (2 * h)

    save_tt(filename, f, L, N)
    
    Return = namedtuple('Return', ['f', 'Dens', 'Vel', 'Temp', 'Frob_norm_iter', 'Frob_norm_RHS'])
    
    S = Return(f, Dens, Vel, Temp, Frob_norm_iter, Frob_norm_RHS)
    
    return S
