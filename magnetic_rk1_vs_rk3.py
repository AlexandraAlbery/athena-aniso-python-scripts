# Compares rk1 vs rk3 timestep for ohmic+ambi+Hall

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import matplotlib.colors as mcolors
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amsthm} \usepackage{amssymb}'


data = pd.read_csv(r"/home/alex/OneDrive/PhD/athena-aniso/Paper/tempdependent_mu_and_gamma.dat",sep='\\s+',header=None)
logT_data = data.loc[:,0]
T_data = 10**logT_data
mu_data = data.loc[:,1]
H2_data = data.loc[:,3]
HI_data = data.loc[:,4]
HII_data = data.loc[:,5]
Prho_data = data.loc[:,6]
X = 0.7381
Y = 0.2485
Z = 0.0134
m_H = 1.6726219e-24   # g
m_e = 9.10938356e-28  # g
e_cgs = 4.80320425e-10  # statC
eV_erg = 1.602176634e-12
ev_K = 8.617e-5
kb = 1.380649e-16
c= 2.99792458e10 # cm/s
e= 4.8e-10
xe_floor = 1e-6

dx = 1
dx_phys = 3.086e18   # cm: cell size for the single-fluid closure check in is_valid
div_4_root3 = 4.0/np.sqrt(3.0)

T = np.logspace(0,9,25,endpoint=True)
rho = np.logspace(-30,-16,15,endpoint=True)
B = np.logspace(-20,4,25,endpoint=True)
logT = np.log10(T)

shape = (np.size(T), np.size(B), np.size(rho))

eta_arr = np.zeros(shape)
eta_AD_arr = np.zeros(shape)
eta_H_arr = np.zeros(shape)

def get_HII(logT_):
    return np.max([np.interp(logT_,logT_data,HII_data),xe_floor/1.2])

def get_ne(logT_, rho_, HII_):
    return np.max([xe_floor*X*rho_/m_H ,1.2*HII_*X*rho_/m_H])

def get_nN(rho_, HII_):
    return (1-HII_)*X*rho_/m_H

def get_Lambda(T_,ne_):
    if(T_*ev_K < 50):
        L = 23.4 - 1.15*np.log10(ne_) + 3.45*np.log10(ev_K*T_)
    else:
        L = 25.3 - 1.15*np.log10(ne_) + 2.3*np.log10(ev_K*T_)
    return L

def get_v_ni(T_, nN_):
    return 0.5 * 1e-9 * nN_ * 0.649*(T_**(0.375))   # m_n/(m_i+m_n) = 1/2 for H+ - H

def get_v_en(T_, nN_):
    Tf = min(T_, 1e4)   # fit turns negative above 3.4e4 K: hold <sigma v> at its 1e4 K value
    return 1e-9 * nN_ * np.sqrt(Tf) * (2.841 + 0.093*np.log10(Tf) + 0.245*np.log10(Tf)**2 -0.089*np.log10(Tf)**3)

def get_v_ei(T_, ne_, L_):
    return 2.91e-6 * ne_ * L_ * ((ev_K*T_)**(-3/2))

def get_sigma(ne_, v_ei_, v_en_):
    return (e_cgs*e_cgs*ne_)/(m_e*(v_ei_ + v_en_))

def get_eta(sigma_):
    return (c*c)/(4*np.pi*sigma_)

def get_etaAD(B_, rho_, HII_, v_in_):
    rho_n = (1.0-HII_)*rho_
    rho_i = HII_*rho_
    eta_AD_ = (B_*B_*rho_n*rho_n)/(4*np.pi*rho_i*v_in_*rho_*rho_)
    if np.isnan(eta_AD_):
        eta_AD_ = 0.0
    if not np.isfinite(eta_AD_):
        eta_AD_ = 0.0
    return eta_AD_

def get_etaH(B_, ne_):
    return (c*B_)/(4*np.pi*e*ne_)

def is_valid(T_, B_, rho_, etaAD_, dx_phys):
    """Braginskii-closure / non-relativistic validity."""
    vA = B_/np.sqrt(4.0*np.pi*rho_)
    if vA >= 0.3*c:                 return False   # (1) non-relativistic MHD
    if kb*T_/(m_e*c*c) >= 0.03:      return False   # (2) non-relativistic electrons
    if etaAD_ >= vA*dx_phys:        return False   # (3) single-fluid ambipolar closure
    return True

dt_rk1 = np.zeros(shape)
dt_rk3 = np.zeros(shape)
dt_ratio = np.zeros(shape)

for i in range(np.size(T)):
    for j in range(np.size(B)):
        for k in range(np.size(rho)):
            HII=get_HII(logT[i])
            ne=get_ne(logT[i],rho[k],HII)
            nN=get_nN(rho[k],HII)
            Lam=get_Lambda(T[i],ne)
            v_in=get_v_ni(T[i],nN)
            v_en=get_v_en(T[i],nN)
            v_ei=get_v_ei(T[i],ne,Lam)
            sigma=get_sigma(ne,v_ei,v_en)
            eta=get_eta(sigma)
            etaAD=get_etaAD(B[j],rho[k],HII,v_in)
            etaH=get_etaH(B[j],ne)
            
            eta_arr[i,j,k] = eta
            eta_AD_arr[i,j,k] = etaAD
            eta_H_arr[i,j,k] = etaH
            
            dt_rk1[i,j,k] = dx*dx*(eta+etaAD)/(2*((eta+etaAD)**2+etaH**2))
            dt_rk3[i,j,k] = dx*dx/(div_4_root3*np.sqrt((eta+etaAD)**2+etaH**2))
            dt_ratio[i,j,k] = dt_rk3[i,j,k]/dt_rk1[i,j,k]

for i in range(np.size(T)):
    for j in range(np.size(B)):
        for k in range(np.size(rho)):
            if not is_valid(T[i], B[j], rho[k], eta_AD_arr[i,j,k], dx_phys):   # this cell's eta_AD, physical cell size
               dt_ratio[i,j,k]   = np.nan
               
print("smallest ratio dt_RK3/dt_RK1:", np.nanmin(dt_ratio))
print("largest ratio dt_RK3/dt_RK1:", np.nanmax(dt_ratio))
 
#%%

logB, logT, logrho = np.log10(B), np.log10(T), np.log10(rho)
nrows, ncols = 5, 3      # 15 rho slices

def pow10_range(data):
    a = np.abs(data[np.isfinite(data)])
    a = a[a > 0]
    if a.size == 0:
        return 1e-300, 1.0
    lo = 10.0**np.floor(np.log10(a.min()))
    hi = 10.0**np.ceil (np.log10(a.max()))
    return lo, hi

def grid_maps(cube, title, cbar_label):
    a = np.abs(cube[np.isfinite(cube) & np.isfinite(cube)])
    a = a[a > 0]
    vmin = 10.0**np.floor(np.log10(a.min()))
    vmax = 10.0**np.ceil (np.log10(a.max()))
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap('turbo').copy(); cm.set_bad('0.6')

    fig, axs = plt.subplots(nrows, ncols, figsize=(11, 15),
                            constrained_layout=True, sharex=True, sharey=True)
    axs = axs.ravel()

    im = None
    for k in range(np.size(rho)):
        data = cube[:, :, k].copy()
        fin = np.isfinite(data)                            # inf (kappa_Jw=0) -> top
        if fin.any(): data[np.isposinf(data)] = data[fin].max()
        im = axs[k].pcolormesh(logB, logT, data, cmap=cm, norm=norm, shading='nearest')
        axs[k].set_title(rf'$\rho=10^{{{np.log10(rho[k]):.0f}}}$ g cm$^{{-3}}$', fontsize=9)
    for k in range(np.size(rho), nrows*ncols):
        axs[k].set_visible(False)

    for a_ in axs[:np.size(rho)]:
        a_.set_xlabel(r'$\log_{10}(B/\mathrm{G})$')
    for r in range(nrows):
        axs[r*ncols].set_ylabel(r'$\log_{10}(T/\mathrm{K})$')

    fig.suptitle(title)
    fig.colorbar(im, ax=axs.tolist(), extend='both', shrink=0.6, label=cbar_label)
    plt.show()

grid_maps(dt_ratio,
          r'',
          r'$\Delta t_{\rm RK3}/\Delta t_{\rm RK1}$')