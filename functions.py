
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')

import numpy as np
from matplotlib import pyplot as plt
import time
import tt

def rs(f1, f2, f, vx_, N):
    for j in range(N):
        if (vx_[j] > 0.):
            f[j, :, :] = vx_[j] * f1[j, :, :]
        else:
            f[j, :, :] = vx_[j] * f2[j, :, :]
    return f
    
def F_m(vx, vy, vz, T, n, p):
    return n * ((1. / (2. * np.pi * p.Rg * T)) ** (3. / 2.)) * (np.exp(-(vx*vx + vy*vy + vz*vz) / (2. * p.Rg * T)))

def J(f, vmax, N, p):

    hv = 2. * vmax / N
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, N)

    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')

    assert np.all(vx[:,0,0] == vx_)
    assert np.all(vy[0,:,0] == vx_)
    assert np.all(vz[0,0,:] == vx_)

    n = (hv ** 3) * np.sum(f)

    ux = (1. / n) * (hv ** 3) * np.sum(vx * f)
    uy = (1. / n) * (hv ** 3) * np.sum(vy * f)
    uz = (1. / n) * (hv ** 3) * np.sum(vz * f)
    
    v2 = vx*vx + vy*vy + vz*vz
    u2 = ux*ux + uy*uy + uz*uz
    
    T = (1. / (3. * n * p.Rg)) * ((hv ** 3) * np.sum(v2 * f) - n * u2)

    Vx = vx - ux
    Vy = vy - uy
    Vz = vz - uz

    rho = p.m * n

    P = rho * p.Rg * T

    cx = Vx / ((2. * p.Rg * T) ** (1. / 2.))
    cy = Vy / ((2. * p.Rg * T) ** (1. / 2.))
    cz = Vz / ((2. * p.Rg * T) ** (1. / 2.))
    
    c2 = cx*cx + cy*cy + cz*cz

    Sx = (1. / n) * (hv ** 3) * np.sum(cx * c2 * f)
    Sy = (1. / n) * (hv ** 3) * np.sum(cy * c2 * f)
    Sz = (1. / n) * (hv ** 3) * np.sum(cz * c2 * f)

    mu = p.mu_0 * ((p.T_0 + p.C) / (T + p.C)) * ((T / p.T_0) ** (3. / 2.))

    f_plus = F_m(Vx, Vy, Vz, T, n, p) * (1. + (4. / 5.) * (1. - p.Pr) * (cx*Sx + cy*Sy + cz*Sz) * (c2 - (5. / 2.)))

    J = (f_plus - f) * (P / mu)
    
    return J, n, ux, T

def solver(x_l, x_r, L, Tau, CFL, vmax, N, n_l, u_l, T_l, p):

    h = (x_r - x_l) / L 
    tau = h * CFL / vmax / 10
    
    x = np.linspace(x_l+h/2, x_r-h/2, L)
    
    t = 0.
    
    hv = 2. * vmax / N
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, N)
    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')
    
    vx_l = np.zeros((N, N, N))
    vx_r = np.zeros((N, N, N))
    
    for i in np.ndindex(vx.shape):
        if vx[i] > 0:
            vx_l[i] = vx[i]
            vx_r[i] = 0.
        else:
            vx_l[i] = 0.
            vx_r[i] = vx[i]
    
    M = u_l / ((p.g * p.Rg * T_l) ** .5)
    
    n_r = (p.g + 1.) * M * M / ((p.g - 1.) * M * M + 2.) * n_l
    u_r = ((p.g - 1.) * M * M + 2.) / ((p.g + 1.) * M * M) * u_l
    T_r = (2. * p.g * M * M - (p.g - 1.)) * ((p.g - 1.) * M * M + 2.) / ((p.g + 1) ** 2 * M * M) * T_l
    
    
    F_l = F_m(vx-u_l, vy, vz, T_l, n_l, p)
    F_r = F_m(vx-u_r, vy, vz, T_r, n_r, p)
    
    
    # initial condition 
    f = np.zeros((L, N, N, N))
    for i in range(L/2+1):
        f[i, :, :, :] = F_l
    for i in range(L/2+1, L):
        f[i, :, :, :] = F_r
    
    slope = np.zeros((L, N, N, N))
    tmp = np.zeros((L, N, N, N))
    f_l = np.zeros((L+1, N, N, N))
    f_r = np.zeros((L+1, N, N, N))
    Flow = np.zeros((L+1, N, N, N))
    RHS = np.zeros((L, N, N, N))
    j = np.zeros((L, N, N, N))
    
    Dens = np.zeros(L)
    Vel = np.zeros(L)
    Temp = np.zeros(L)
    
    Frob_norm = np.array([])
    C_norm = np.array([])
    
    t1 = time.clock()
    
    while(t < Tau*tau):
        t += tau
        # boundary condition
        f_l[0, :, :, :] = F_l
        f_r[L, :, :, :] = F_r
        # reconstruction
        # compute slopes
        
        for i in range(1, L-1):
            slope[i, :, :, :] = h * (f[i+1, :, :, :] - 2 * f[i, :, :, :] + f[i-1, :, :, :])
#            slope[i, :, :, :] = h * minmod(f[i+1, :, :, :] - f[i, :, :, :], f[i, :, :, :] - f[i-1, :, :, :])
            
        for i in range(L):
            f_r[i, :, :, :] = f[i, :, :, :] - (h / 2) * slope[i, :, :, :]
        
        for i in range(1, L+1):
            f_l[i, :, :, :] = f[i-1, :, :, :] + (h / 2) * slope[i-1, :, :, :]
        
        # riemann solver - compute fluxes
        for i in range(L+1):
            Flow[i, :, :, :] = rs(f_l[i, :, :, :], f_r[i, :, :, :], Flow[i, :, :, :], vx_, N)
            
        
            
        # compute RHS
        for i in range(L):
            RHS[i, :, :, :] = (- Flow[i+1, :, :, :] + Flow[i, :, :, :]) / h + J(f[i, :, :, :], vmax, N, p)[0]
            Frob_norm = np.append(Frob_norm, np.linalg.norm(RHS))
            C_norm = np.append(C_norm, np.max(np.absolute(np.ravel(RHS))))

        # update values
        for i in range(L):
            tmp[i, :, :, :] = f[i, :, :, :] + tau * RHS[i, :, :, :]
                
        f = tmp
        
#        print np.linalg.norm(RHS)
#        print np.max(np.absolute((j)))
        
     
        
    t2 = time.clock() - t1
    Dens = np.zeros(L)
    Vel = np.zeros(L)
    Temp = np.zeros(L)
    
    for i in range(L):
        Dens[i] = J(f[i, :, :, :], vmax, N, p)[1]
        Vel[i] = J(f[i, :, :, :], vmax, N, p)[2]
        Temp[i] = J(f[i, :, :, :], vmax, N, p)[3]
        
    print "time =", t2
    
    fig, ax = plt.subplots(figsize = (20,10))
    line, = ax.semilogy(Frob_norm)
    line.set_label('Frob_norm')
    line, = ax.semilogy(C_norm)
    line.set_label('C_norm')
    ax.legend()
    
    return f, Dens, Vel, Temp

    
def rs_tt(f1, f2, vx_l, vx_r):
    return vx_l * f1 + vx_r * f2

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
    
    return J, n, ux, T

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

def solver_tt(x_l, x_r, L, Tau, CFL, vmax, N, n_l, u_l, T_l, r, p):
    
    h = (x_r - x_l) / L 
    tau = h * CFL / vmax / 10
    
    x = np.linspace(x_l+h/2, x_r-h/2, L)
    
    t = 0.
    
    hv = 2. * vmax / N
    vx_ = np.linspace(-vmax+hv/2, vmax-hv/2, N)
    vx, vy, vz = np.meshgrid(vx_, vx_, vx_, indexing='ij')
    
    vx_l = np.zeros((N, N, N))
    vx_r = np.zeros((N, N, N))
    
    for i in np.ndindex(vx.shape):
        if vx[i] > 0:
            vx_l[i] = vx[i]
        else:
            vx_l[i] = 0.
            
    vx_r = vx - vx_l
            
    vx_l = tt.tensor(vx_l)
    vx_r = tt.tensor(vx_r)
    
    vx_tt = tt.tensor(vx)
    vy_tt = tt.tensor(vy)
    vz_tt = tt.tensor(vz)
    
    M = u_l / ((p.g * p.Rg * T_l) ** .5)
    
    n_r = (p.g + 1.) * M * M / ((p.g - 1.) * M * M + 2.) * n_l
    u_r = ((p.g - 1.) * M * M + 2.) / ((p.g + 1.) * M * M) * u_l
    T_r = (2. * p.g * M * M - (p.g - 1.)) * ((p.g - 1.) * M * M + 2.) / ((p.g + 1) ** 2 * M * M) * T_l
    
    
    F_l = tt.tensor(F_m(vx-u_l, vy, vz, T_l, n_l, p))
    F_r = tt.tensor(F_m(vx-u_r, vy, vz, T_r, n_r, p))
    
#    vx = tt.tensor(vx)
#    vy = tt.tensor(vy)
#    vz = tt.tensor(vz)
    
    # initial condition 
#    f = list(F_l for i in range(L/2+1))
#    f = f.extend(list(F_r for i in range(L/2+1, L)))

    f = list(tt.tensor(np.zeros((N, N, N))) for i in range(L))
    
    for i in range(L/2+1):
        f[i] = F_l
    for i in range(L/2+1, L):
        f[i] = F_r
        
#    for i in range(L):
#        f[i] = tt.tensor(f[i])
    
    slope = list(0. * tt.ones((N,N,N)) for i in range(L))
    tmp = list(0. * tt.ones((N,N,N)) for i in range(L))
    f_l = list(0. * tt.ones((N,N,N)) for i in range(L+1))
    f_r = list(0. * tt.ones((N,N,N)) for i in range(L+1))
    Flow = list(0. * tt.ones((N,N,N)) for i in range(L+1))
    RHS = list(0. * tt.ones((N,N,N)) for i in range(L))
    J_ = list(0. * tt.ones((N,N,N)) for i in range(L))
    

    Frob_norm = np.zeros(L)
    Frob_norm_iter = np.array([])
#    C_norm_iter = np.array([])
        
    t1 = time.clock()
    
    while(t < Tau*tau):
        t += tau
        # boundary condition
        f_l[0] = F_l
        f_r[L] = F_r
        # reconstruction
        # compute slopes
        
        for i in range(1, L-1):
            slope[i] = h * (f[i+1] - 2 * f[i] + f[i-1])
            slope[i] = slope[i].round(r)
#            print slope[i] + f[i]
#            slope[i] = slope[i].round(rmax=150)
            # insert ROUNDING!!!
#            slope[i, :, :, :] = h * minmod(f[i+1, :, :, :] - f[i, :, :, :], f[i, :, :, :] - f[i-1, :, :, :])
            
        for i in range(L):
            f_r[i] = f[i] - slope[i] * (h / 2)
            f_r[i] = f_r[i].round(r)
        
        for i in range(1, L+1):
            f_l[i] = f[i-1] + (h / 2) * slope[i-1]
            f_l[i] = f_l[i].round(r)
        
        # riemann solver - compute fluxes
        for i in range(L+1):
            Flow[i] = rs_tt(f_l[i], f_r[i], vx_l, vx_r)
            Flow[i] = Flow[i].round(r)
            
        # compute RHS
        for i in range(L):
            J_[i] = J_tt(f[i], vx, vy, vz, vx_tt, vy_tt, vz_tt, hv, N, r, p)
            RHS[i] = (- Flow[i+1] + Flow[i]) * (1. / h) + J_[i][0]
            RHS[i] = RHS[i].round(r)
            
        if (((t/tau) % 100) == 1):
            for i in range(L):
                Frob_norm[i] = RHS[i].norm()
                
                
            fig, ax = plt.subplots(figsize = (20,10))
            ax.plot(Frob_norm)

            ax.set(title='RHS frob norm')

            fig.savefig("RHS.png")
            
            
            save_tt('file.txt', f, L, N)
                
        Frob_norm_iter = np.append(Frob_norm_iter, RHS[i].norm())

        # update values
        for i in range(L):
            tmp[i] = f[i] + tau * RHS[i]
            tmp[i] = tmp[i].round(r)
            
        for i in range(L):    
            f[i] = tmp[i]
        
#        print np.linalg.norm(RHS)
#        print np.max(np.absolute((j)))
        
     
        
    t2 = time.clock() - t1
    
    Dens = np.zeros(L)
    Vel = np.zeros(L)
    Temp = np.zeros(L)



    for i in range(L):
        
        Dens[i] = J_[i][1]
        Vel[i] = J_[i][2]
        Temp[i] = J_[i][3]


    print "time =", t2
    
    
    return f, Dens, Vel, Temp, Frob_norm_iter#, C_norm_iter

