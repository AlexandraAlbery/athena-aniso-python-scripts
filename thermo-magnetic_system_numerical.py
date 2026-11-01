# Checks well-posedness and timesteps for thermo-magnetic system

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

data = pd.read_csv(r"./tempdependent_mu_and_gamma.dat",sep='\\s+',header=None)
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
dx_phys = 3.086e18   # cm

dx = 1
dp = 0.5

T = np.logspace(0,9,25,endpoint=True)
rho = np.logspace(-30,-16,15,endpoint=True)
B = np.logspace(-20,4,25,endpoint=True)
logT = np.log10(T)

#%%
def get_HII(logT_):
    return np.max([np.interp(logT_,logT_data,HII_data),xe_floor/1.2])

def get_mu(logT_):
    return np.interp(logT_,logT_data,mu_data)

def get_E_int(logT_, rho_):
    return np.interp(logT_,logT_data,Prho_data)*rho_/((5.0/3.0) - 1.0)

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

def get_v_ii(T_, ni_, L_):
    return 4.80e-8 *ni_ * L_ * ((ev_K*T_)**(-3/2))

def get_v_en(T_, nN_):
    Tf = min(T_, 1e4)   # fit turns negative above 3.4e4 K: hold <sigma v> at its 1e4 K value
    return 1e-9 * nN_ * np.sqrt(Tf) * (2.841 + 0.093*np.log10(Tf) + 0.245*np.log10(Tf)**2 -0.089*np.log10(Tf)**3)

def get_v_ei(T_, ne_, L_):
    return 2.91e-6 * ne_ * L_ * ((ev_K*T_)**(-3/2))

def get_omega_e(B_):
    return e_cgs*B_ /(m_e*c)

def get_beta_e(omega_e_, v_ei_, v_en_):
    return omega_e_/(v_ei_+v_en_)

def get_sigma(ne_, v_ei_, v_en_):
    return (e_cgs*e_cgs*ne_)/(m_e*(v_ei_ + v_en_))

def get_cv(mu_):
    return kb/(mu_*m_H*(2/3))

def get_kappa_iso(T_, rho_, cv_):
    return 2.5e5*np.sqrt(T_/1e4) / (cv_*rho_)

def get_kappa_aniso(T_, ne_, rho_, cv_):
    return ((1.70e11*((T_/1e7)**(5/2)))/(1+0.029*np.log((T_/1e7)*((ne_/(1e-2))**(-1/2)))) ) / (cv_*rho_)

def get_kappa_neutral(HII_, kappa_iso_):
    return (1.0 - HII_)*kappa_iso_

def get_kappa_para(HII_, kappa_aniso_):
    return HII_*kappa_aniso_

def get_kappa_perp(HII_, beta_e_, kappa_aniso_):
    return HII_*(1.0/3.1616)*kappa_aniso_*((4.664*beta_e_*beta_e_ + 11.92)/((beta_e_**4)+14.79*beta_e_*beta_e_ + 3.7703))

def get_kappa_wedge(HII_, beta_e_, kappa_aniso_):
    return HII_*(1.0/3.1616)*kappa_aniso_*beta_e_*((2.5*beta_e_*beta_e_ + 21.67)/((beta_e_**4)+14.79*beta_e_*beta_e_ + 3.7703))

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

def get_kappa_J_para(T_, HII_, rho_, cv_):
    return HII_*0.7110*c*kb*T_/(np.sqrt(4*np.pi)*e_cgs)
def get_kappa_J_perp(T_, HII_, rho_, cv_, beta_e_):
    return HII_*(c*kb*T_/(np.sqrt(4*np.pi)*e_cgs))*(5.101*beta_e_**2 + 2.681)/(beta_e_**4 + 14.79*beta_e_**2 + 3.7703)
def get_kappa_J_wedge(T_, HII_, rho_, cv_, beta_e_):
    return HII_*(c*kb*T_/(np.sqrt(4*np.pi)*e_cgs))*(beta_e_*(1.5*beta_e_**2 + 3.053))/(beta_e_**4 + 14.79*beta_e_**2 + 3.7703)

LAM_TOL = 0

def rk1_cfl(lam, R_rk1):
    lam  = np.asarray(lam).ravel() 
    absl = np.abs(lam)   
    peak = absl.max()    
    if peak == 0.0:        
        return np.inf   
    lam = lam[absl > LAM_TOL*peak]  
    return np.min(-2.0*lam.real / (lam.real**2 + lam.imag**2))

def rk3_cfl(lam, R):
    lam  = np.asarray(lam).ravel()
    absl = np.abs(lam)
    peak = absl.max()
    if peak == 0.0:
        return np.inf                                        
    keep = absl > LAM_TOL*peak                               
    lam, absl = lam[keep], absl[keep]                       
    unit = lam/absl                           
    lo = np.zeros(lam.shape); hi = np.full(lam.shape, 4.0)   # s = dt*|lambda|
    for _ in range(60): hi = np.where(np.abs(R(hi*unit)) <= 1.0, hi*2.0, hi)
    for _ in range(70):
        m = 0.5*(lo+hi); ok = np.abs(R(m*unit)) <= 1.0
        lo = np.where(ok, m, lo); hi = np.where(ok, hi, m)
    return np.min(lo/absl)

def rk3_dt_each(lam, R):
    """rk3_cfl without the final min: the RK3 limit for each eigenvalue (inf for lambda = 0)."""
    lam  = np.asarray(lam).ravel()
    absl = np.abs(lam)
    out  = np.full(lam.shape, np.inf)
    keep = absl > 0.0
    unit = lam[keep]/absl[keep]
    lo = np.zeros(unit.shape); hi = np.full(unit.shape, 4.0)
    for _ in range(60): hi = np.where(np.abs(R(hi*unit)) <= 1.0, hi*2.0, hi)
    for _ in range(70):
        m = 0.5*(lo+hi); ok = np.abs(R(m*unit)) <= 1.0
        lo = np.where(ok, m, lo); hi = np.where(ok, hi, m)
    out[keep] = lo/absl[keep]
    return out

def is_valid(T_, B_, rho_, etaAD_, dx_phys):
    """Braginskii-closure / non-relativistic validity."""
    vA = B_/np.sqrt(4.0*np.pi*rho_)
    if vA >= 0.3*c:                 return False   # (1) non-relativistic MHD
    if kb*T_/(m_e*c*c) >= 0.03:      return False   # (2) non-relativistic electrons
    if etaAD_ >= vA*dx_phys:        return False   # (3) single-fluid ambipolar closure
    return True

n = 20

dx = 1.0
dp = 0.5
root3 = np.sqrt(3.0)
div_4_root3 = 4.0/np.sqrt(3.0)

R_rk1 = lambda z: 1.0 + z                          # forward Euler
R_rk2 = lambda z: 1.0 + z + 0.5*z*z                # SSP-RK2
R_rk3 = lambda z: 1.0 + z + 0.5*z*z + z*z*z/6.0    # SSP-RK3

#%%
# ============================================================================

TH1 = np.pi/2
PH1 = 0;
## WARNING: using the actual trig functions causes extreme floating point errors
## bx = 0, by or bz = 1 for worst-case ill-posed and perp regime
## bx =1, by=bz=0 for parallel regime
BX1 = 0.0 # np.cos(TH1)
BY1 = 1.0 #(np.sin(TH1)*np.cos(PH1))
BZ1 = 0.0 # (np.sin(TH1)*np.sin(PH1))
U1  = BX1*BX1                      # u = cos^2(theta)
FAC1 = 1.0                         # sin^2(psi) = 1
FACW1 = 1.0                        #  sin(2 psi) = 1
diff = -4.0/(dx*dx)
diff_w = 2.0/(dx*dp)

SCAN_ORIENTATION = False   # True: scan u = cos^2(theta) and psi = k dx/2
U_SCAN   = np.linspace(0.0, 1.0, 21)
PSI_SCAN = [(1.0, 0.0), (np.sin(3*np.pi/8)**2, np.sqrt(0.5)), (0.5, 1.0), (np.sin(np.pi/8)**2, np.sqrt(0.5))]  # (sin^2 psi, sin 2psi)

shape = (np.size(T), np.size(B), np.size(rho))

dt_bound_1 = np.zeros(shape)
dt_num_1   = np.zeros(shape)
dt_diff_1  = np.zeros(shape)
dt_ratio  = np.zeros(shape)
wp_1       = np.zeros(shape)       # analytic well-posed margin (scalar, worst orientation)
wp_num_1   = np.zeros(shape)       # numerical: top eigenvalue of physical Hermitian part
u_bind_1   = np.zeros(shape)       # u = cos^2(theta) of the mode that sets dt_num (-1: all u within round-off)
psi_bind_1 = np.zeros(shape)       # psi = k dx/2 of that mode
lam_max_1  = np.zeros(shape)       # largest |eigenvalue|, the scale of the round-off in wp_num_1

kappa_neutral_arr = np.zeros(shape)
kappa_para_arr = np.zeros(shape)
kappa_perp_arr = np.zeros(shape)
kappa_wedge_arr = np.zeros(shape)

eta_arr = np.zeros(shape)
eta_AD_arr = np.zeros(shape)
eta_H_arr = np.zeros(shape)

kappa_J_para_arr = np.zeros(shape)
kappa_J_perp_arr = np.zeros(shape)
kappa_J_wedge_arr = np.zeros(shape) 
M = np.zeros((3,3), complex)

for i in range(np.size(T)):
    for j in range(np.size(B)):
        for k in range(np.size(rho)):
            HII=get_HII(logT[i])
            mu=get_mu(logT[i])
            E_int=get_E_int(logT[i],rho[k])
            ne=get_ne(logT[i],rho[k],HII)
            nN=get_nN(rho[k],HII)
            Lam=get_Lambda(T[i],ne)
            v_in=get_v_ni(T[i],nN)
            v_en=get_v_en(T[i],nN)
            v_ei=get_v_ei(T[i],ne,Lam)
            omega_e=get_omega_e(B[j])
            beta_e=get_beta_e(omega_e,v_ei,v_en)
            sigma=get_sigma(ne,v_ei,v_en)
            cv=get_cv(mu)
            kappa_iso=get_kappa_iso(T[i],rho[k],cv)
            kappa_aniso=get_kappa_aniso(T[i],ne,rho[k],cv)
            kappa_neutral=get_kappa_neutral(HII,kappa_iso)
            kappa_para=get_kappa_para(HII,kappa_aniso)
            kappa_perp=get_kappa_perp(HII,beta_e,kappa_aniso)
            kappa_wedge=get_kappa_wedge(HII,beta_e,kappa_aniso)
            eta=get_eta(sigma); etaAD=get_etaAD(B[j],rho[k],HII,v_in); etaH=get_etaH(B[j],ne)
            kappa_J_para=get_kappa_J_para(T[i],HII,rho[k],cv)
            kappa_J_perp=get_kappa_J_perp(T[i],HII,rho[k],cv,beta_e)
            kappa_J_wedge=get_kappa_J_wedge(T[i],HII,rho[k],cv,beta_e)
            
            kappa_neutral_arr[i,j,k] = kappa_neutral
            kappa_para_arr[i,j,k] = kappa_para
            kappa_perp_arr[i,j,k] = kappa_perp
            kappa_wedge_arr[i,j,k] = kappa_wedge
            
            eta_arr[i,j,k] = eta
            eta_AD_arr[i,j,k] = etaAD
            eta_H_arr[i,j,k] = etaH
            
            kappa_J_para_arr[i,j,k] = kappa_J_para/np.sqrt(E_int)
            kappa_J_perp_arr[i,j,k] = kappa_J_perp/np.sqrt(E_int)
            kappa_J_wedge_arr[i,j,k] =  kappa_J_wedge/np.sqrt(E_int)

            kappa_0   = kappa_neutral + kappa_perp
            kappa_w   = kappa_wedge
            d_kappa_J = kappa_J_perp - kappa_J_para
            kappa_J_w = kappa_J_wedge
            gm1 = (5.0/3.0) - 1.0      # gamma-1 used in the conduction operator (matches get_cv)

            # ---- analytic bound: unsplit thermo-magnetic timestep of Section 3 ----
            # dt_TM = (1/dt_TM,R + 1/dt_kappa,wedge + 1/dt_HJ)^-1
            m_mag = eta + etaAD
            a_0 = kappa_0                                 # (gamma-1) kappa_0
            a_1 = kappa_neutral + kappa_para              # (gamma-1)(kappa_0 + Delta kappa)
            lam_R0 = 0.5*(a_0 + m_mag) + np.sqrt(0.25*(a_0 - m_mag)**2 + kappa_J_w**2/E_int)
            lam_R1 = max(a_1, m_mag)
            dt_TMR = (dx*dx)/(2.0*max(lam_R0, lam_R1))    # both endpoints u = 0, 1
            if d_kappa_J != 0.0 and etaH**2*E_int <= d_kappa_J**2:   # interior u* in [0,1]
                lam_HJ = 0.5*(etaH**2*np.sqrt(E_int)/np.abs(d_kappa_J) + np.abs(d_kappa_J)/np.sqrt(E_int))
            else:
                lam_HJ = etaH
            dt_HJ = (dx*dx)/(div_4_root3*lam_HJ)
            dt_kw = root3*(dx*dp)/(2.0*kappa_w)          # fac_RL = sqrt(3)/2, i.e. sqrt(3)/(diff_w kappa_w)
            dt_bound_1[i,j,k] = 1.0/(1.0/dt_TMR + 1.0/dt_kw + 1.0/dt_HJ)

            # ---- amplification-factor spectrum at psi=pi/2 (FAC=1, FACW=0), or over the scan ----
            cases = ([(np.sqrt(u_), np.sqrt(1.0 - u_), 0.0, f_, fw_) for u_ in U_SCAN for f_, fw_ in PSI_SCAN]
                     if SCAN_ORIENTATION else [(BX1, BY1, BZ1, FAC1, 0.0)])
            lam = []
            for bx, by, bz, fac, facw in cases:   # loop names, so BX1.. keep their values
                DKAP = (kappa_para - kappa_perp)*bx*bx
                M[0,0] = fac*diff*(kappa_0 + DKAP) + 1j*facw*diff_w*kappa_w   # Righi-Leduc ~ sin(2 psi)
                M[0,1] = fac*diff*(-d_kappa_J*bx*bz + kappa_J_w*by)
                M[0,2] = fac*diff*( d_kappa_J*bx*by + kappa_J_w*bz)
                M[1,0] = fac*diff/E_int*( d_kappa_J*bx*bz + kappa_J_w*by)
                M[1,1] = fac*diff*(eta + etaAD*(1.0-bz*bz))
                DAD = fac*diff*etaAD*by*bz
                DH = -fac*diff*etaH*bx
                M[1,2] = DAD - DH
                M[2,0] = fac*diff/E_int*(-d_kappa_J*bx*by + kappa_J_w*bz)
                M[2,1] = DAD + DH
                M[2,2] = fac*diff*(eta + etaAD*(1.0-by*by))
                lam.append(np.linalg.eigvals(M))
            lam = np.concatenate(lam)
            dtc = rk3_dt_each(lam, R_rk3).reshape(-1, 3).min(axis=1)   # limit per case, 3 eigenvalues each
            dt_rk3 = dtc.min()
            c_min = np.argmin(dtc)
            u_near = {cases[c][0]**2 for c in np.flatnonzero(dtc <= dt_rk3*(1.0 + 1e-9))}
            u_bind_1[i,j,k] = -1.0 if (SCAN_ORIENTATION and len(u_near) == len(U_SCAN)) else cases[c_min][0]**2
            psi_bind_1[i,j,k] = np.arcsin(np.sqrt(cases[c_min][3]))
            dt_num_1[i,j,k]  = dt_rk3
            dt_diff_1[i,j,k] = dt_bound_1[i,j,k] - dt_num_1[i,j,k]
            dt_ratio[i,j,k] = dt_bound_1[i,j,k] / dt_num_1[i,j,k] 
            
            wp_num_1[i,j,k] = np.max(lam.real)           # top eigenvalue; >0 => ill-posed
            lam_max_1[i,j,k] = np.max(np.abs(lam))

            # analytic scalar margin (worst orientation u=0): >0 is well posed
            denom = kappa_J_w**2/E_int
            wp_1[i,j,k] = np.log10(kappa_0*(eta+etaAD)/denom) if denom > 0 else np.inf


#%%        
print("\norientation:", "scan over u = cos^2(theta) and psi" if SCAN_ORIENTATION else f"b = ({BX1}, {BY1}, {BZ1}) at psi = pi/2")
print("finite dt_num:", np.all(~np.isnan(dt_num_1)))
print("bound violations (dt_bound > dt_num):", int(np.sum(dt_ratio > 1.0 + 1e-12)))
# In the parallel case floating point errors cause the difference between doubles to round to exactly 1024.0. These are not actual violations
print("ill-posed (analytic, margin<0):", int(np.sum(wp_1 < 0.0)))
print("ill-posed (numerical, top eig of H>0):", int( np.sum( np.real(wp_num_1) > 1e-12*lam_max_1 )  )  )   # above round-off
print("most conservative ratio dt_bound/dt_num:", np.nanmin(dt_ratio))
print("least conservative ratio dt_bound/dt_num:", np.nanmax(dt_ratio))

#%%
for i in range(np.size(T)):
    for j in range(np.size(B)):
        for k in range(np.size(rho)):
            if not is_valid(T[i], B[j], rho[k], eta_AD_arr[i,j,k], dx_phys):
               dt_bound_1[i,j,k] = np.nan
               dt_num_1[i,j,k]   = np.nan
               dt_diff_1[i,j,k]  = np.nan
               dt_ratio[i,j,k]   = np.nan
               wp_1[i,j,k]       = np.nan
               wp_num_1[i,j,k]   = np.nan
               u_bind_1[i,j,k]   = np.nan
               psi_bind_1[i,j,k] = np.nan
    
if SCAN_ORIENTATION:
    v = np.isfinite(u_bind_1)
    print("binding mode, fraction of valid cells: orientation-independent %.3f, u = 1 %.3f, other u %.3f"
          % (np.mean(u_bind_1[v] < 0.0), np.mean(u_bind_1[v] == 1.0), np.mean((u_bind_1[v] >= 0.0) & (u_bind_1[v] < 1.0))))
    print("binding psi/pi:", np.unique(np.round(psi_bind_1[v]/np.pi, 3)))

    #%%
# ---- combined 5x3 panels: dt margin and well-posedness, one shared colorbar each ----

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

def grid_maps(cube, title, cbar_label, norm=None, under=None, extend='both'):
    if norm is None:
        a = np.abs(cube[np.isfinite(cube) & np.isfinite(cube)])
        a = a[a > 0]
        if a.max() < 10.0*a.min():                         # under a decade: linear, so the variation shows
            norm = mcolors.Normalize(vmin=a.min(), vmax=a.max())
        else:
            vmin = 10.0**np.floor(np.log10(a.min()))
            vmax = 10.0**np.ceil (np.log10(a.max()))
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap('turbo').copy(); cm.set_bad('0.6')
    if under is not None: cm.set_under(under)              # colour below the norm, e.g. white for u_bind = -1

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
    fig.colorbar(im, ax=axs.tolist(), extend=extend, shrink=0.6, label=cbar_label)
    plt.show()

grid_maps(dt_ratio,
          r'',
          r'$\Delta t_{\rm bound}/\Delta t_{\rm num}$')

grid_maps(wp_1,
          r'',
          r'$\log_{10}[\kappa_0(\eta+\eta_{AD})\varepsilon/\kappa_{Jw}^2]$')

grid_maps(-wp_num_1,
          r'',
          r'$\log_{10}[-\text{Re}(\lambda_{M,~\kappa,\eta,J}])$')

if SCAN_ORIENTATION:
    grid_maps(u_bind_1,
              r'',
              r'$u=\cos^{2}\theta$ of the binding mode', norm=mcolors.Normalize(0.0, 1.0),
              under='w', extend='min')