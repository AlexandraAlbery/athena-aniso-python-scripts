# Tests timesteps for 

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

# ---------------- constants (cgs) ----------------
m_H=1.6726219e-24; m_e=9.10938356e-28; e_cgs=4.80320425e-10
kb=1.380649e-16; c_light=2.99792458e10; ev_K=8.617e-5
X_H=0.7381; xe_floor=1e-6
dx=1.0;
dx_phys = 3.086e18   # cm: cell size for the single-fluid closure check in is_valid
c= 2.99792458e10 # cm/s

data = pd.read_csv("./tempdependent_mu_and_gamma.dat", sep=r'\s+', header=None)
logT_d=data.loc[:,0]; mu_d=data.loc[:,1]; H2_d=data.loc[:,3]; HII_d=data.loc[:,5]

T=np.logspace(0,9,25); rho=np.logspace(-30,-16,15); B=np.logspace(-20,4,25)
logT=np.log10(T)

# ---------------- plasma getters ----------------
def get_HII(lT): return max(np.interp(lT,logT_d,HII_d), xe_floor/1.2)   # floor
def get_H2(lT):  return np.interp(lT,logT_d,H2_d)
def get_ne(lT,r,HII): return max(xe_floor*X_H*r/m_H, 1.2*HII*X_H*r/m_H)
def get_nN(r,HII):    return (1.0-HII)*X_H*r/m_H
def get_Lambda(T_,ne_):
    if T_*ev_K < 50: return 23.4 - 1.15*np.log10(ne_) + 3.45*np.log10(ev_K*T_)
    else:            return 25.3 - 1.15*np.log10(ne_) + 2.3 *np.log10(ev_K*T_)
def get_nu_ii(T_,ni_,L_): return 4.80e-8*ni_*L_*((ev_K*T_)**-1.5)   # NRL: mu = m_i/m_p = 1 already in 4.80e-8
def get_nu_in(T_,nN_):    return 0.5*1e-9*nN_*0.649*(T_**0.375)   # m_n/(m_i+m_n) = 1/2 for H+ - H
def get_omega_ci(B_):     return e_cgs*B_/(m_H*c_light)

# ---------------- bulk viscosity (H2 relaxation) ----------------
def c_rot_H2(T_, th_rot=85.3, Jmax=40):
    c=0.0
    for par,fr in ((0,0.25),(1,0.75)):
        J=np.arange(par,Jmax,2); E=th_rot*J*(J+1)
        w=(2*J+1)*np.exp(-(E-E[0])/T_); Z=w.sum()
        Em=(w*E).sum()/Z; E2=(w*E*E).sum()/Z
        c+=fr*(E2-Em*Em)/T_**2
    return c

def get_zeta_over_rho(T_, lT, r):
    w2=get_H2(lT)
    if w2<=0.0: return 0.0
    crot=c_rot_H2(T_); xv=5984.0/T_
    cvib=xv*xv*np.exp(-xv)/np.expm1(-xv)**2
    gm1=1.0/(1.5+crot+cvib)
    ptr=(1.822e-4+4.64e-6*T_)*10.0                              # Pa s -> poise
    ptv=10.0**(34.74*T_**(-1.0/3.0)-8.686)*1.01325e6            # atm s -> poise
    return w2*gm1*gm1*(crot*ptr + cvib*ptv)/r

# ---------------- shear viscosities ----------------
def get_nu_neu(T_,r): return (8.76e-5*(T_/300.0)**1.5*372.0/(T_+72.0))/r
def get_nu_ion(T_,r,L_): return (2.21e-15*T_**2.5/L_)/r
def brag_perp1(nu_ion,bi):  return nu_ion*(4.8*bi**2+2.23)/(0.96*(16*bi**4+16.12*bi**2+2.33))      # eta_1
def brag_perp2(nu_ion,bi):  return nu_ion*(1.2*bi**2+2.23)/(0.96*4*(bi**4+4.03*bi**2+2.33))        # eta_2/4
def brag_wedge1(nu_ion,bi): return nu_ion*bi*(8*bi**2+4.76)/(0.96*(16*bi**4+16.12*bi**2+2.33))     # eta_3
def brag_wedge2(nu_ion,bi): return nu_ion*bi*(bi**2+2.38)/(0.96*2*(bi**4+4.03*bi**2+2.33))         # eta_4/2

# ---------------- viscous operator (verified vs paper) ----------------
def build_visc(b, Z, X, Vn, VP, Vp1, Vp2, Vw1, Vw2):
    """Matrix acting on (v_x'',v_y'',v_z''). Parabolic part symmetric,
       wedge part skew — both verified against the analytic entries."""
    b=np.asarray(b,float); I3=np.eye(3)
    Bm=np.outer(b,b); Ip=I3-Bm; bx,by,bz=b
    bc=np.array([[0,-bz,by],[bz,0,-bx],[-by,bx,0]])
    M=np.zeros((3,3))
    ddv=np.zeros(3);
    gV=np.zeros((3,3));
    for jc in range(3):
        ddv = np.zeros(3)
        gV  = np.zeros((3,3))
        ddv[jc]=1.0
        gV[0,:]=ddv
        W=gV+gV.T-(2.0/3.0)*ddv[0]*I3
        bWb=b@W@b
        Pi=(Z*ddv[0]*I3 + (1-X)*Vn*W
            + 1.5*X*VP*(Bm-I3/3.0)*bWb
            + X*Vp1*(Ip@W@Ip + 0.5*Ip*bWb)
            + 4*X*Vp2*(Ip@W@Bm + Bm@W@Ip)
            - 0.5*Vw1*X*(bc@W@Ip - Ip@W@bc)
            - 2*Vw2*X*(bc@W@Bm - Bm@W@bc))
        M[:,jc]=Pi[0,:]
    return M

# ---------------- analytic lambda_max (for dt_bound) ----------------
def lam_A_max(Z,X,Vn,VP,Vp1,Vp2):
    def lam_plus(u):
        Tr = Z + (7.0/3.0)*(1-X)*Vn + X*(VP*(u+1.0/3.0) + Vp1*(1-u) + 4*Vp2)
        Det=(Z*(1-X)*Vn + (4.0/3.0)*(1-X)**2*Vn**2
             + X*( Z*((3*VP+Vp1)*u*(1-u) + 4*Vp2*(2*u-1)**2)
                 + (1-X)*Vn*(3*VP*(u-1.0/3.0)**2 + Vp1*(1-u)**2 + 16*Vp2*u*(1-u))
                 + (4.0/3.0)*(1-X)*Vn*((3*VP+Vp1)*u*(1-u) + 4*Vp2*(2*u-1)**2))
             + (4.0/3.0)*X**2*(VP*Vp1*u*(1-u) + VP*Vp2*(u+1)**2 + 3*Vp1*Vp2*(1-u)**2))
        return 0.5*(Tr + np.sqrt(max(Tr*Tr-4*Det, 0.0)))
    cands=[0.0,1.0]
    d1=3*VP+Vp1-16*Vp2
    if d1!=0:
        us=(VP+Vp1-8*Vp2)/d1
        if 0.0<=us<=1.0: cands.append(us)
    d2=-4*Vn*X+4*Vn+VP*X+3*Vp1*X+12*Z
    if d2!=0:
        us=(-2*Vn*X+2*Vn-VP*X+3*Vp1*X+6*Z)/d2
        if 0.0<=us<=1.0: cands.append(us)
    return max(lam_plus(u) for u in cands)

def lam_C_max(X,Vw1,Vw2):
    lam = max(Vw1, 2.0*Vw2)                                       # endpoints u = 0, 1
    if Vw1 > 0.0 and Vw2 <= Vw1 <= (1.0 + np.sqrt(3.0))*Vw2:     # interior u* in [0,1]
        lam = max(lam, 2.0*np.sqrt(Vw2**3*(2.0*Vw1 + Vw2)/(Vw1*(4.0*Vw2 - Vw1))))
    return X*lam

# ---------------- RK3 stability limit ----------------
R_rk3 = lambda z: 1.0 + z + 0.5*z*z + z*z*z/6.0
LAM_TOL=1e-12   # drop |lam| < 1e-12*peak
def rk3_cfl(lam, R):
    lam=np.asarray(lam).ravel()
    absl=np.abs(lam)
    peak=absl.max()
    if peak==0.0: return np.inf
    keep=absl>LAM_TOL*peak; lam,absl=lam[keep],absl[keep]
    unit=lam/absl
    lo=np.zeros(lam.shape); hi=np.full(lam.shape,4.0)
    for _ in range(60): hi=np.where(np.abs(R(hi*unit))<=1.0, hi*2.0, hi)
    for _ in range(70):
        m=0.5*(lo+hi); ok=np.abs(R(m*unit))<=1.0
        lo=np.where(ok,m,lo); hi=np.where(ok,hi,m)
    return np.min(lo/absl)

def is_valid(T_, B_, rho_, etaAD_, dx_phys):
    """Braginskii-closure / non-relativistic validity."""
    vA = B_/np.sqrt(4.0*np.pi*rho_)
    if vA >= 0.3*c:                 return False   # (1) non-relativistic MHD
    if kb*T_/(m_e*c*c) >= 0.03:      return False   # (2) non-relativistic electrons
    if etaAD_ >= vA*dx_phys:        return False   # (3) single-fluid ambipolar closure
    return True

# ---------------- angle grid (psi = pi/2 fixed) ----------------
SCAN_ORIENTATION = False    # False: the single orientation (BX, BY, BZ) below
U_SCAN = np.linspace(0.0, 1.0, 41)   # u = cos^2(theta), b = (sqrt(u), sqrt(1-u), 0): exact zeros at the ends
BX = 1.0
BY = 0.0
BZ = 0.0
diff=-4.0/(dx*dx)                       # sin^2(psi)=1

shape=(T.size,B.size,rho.size)
dt_bound=np.zeros(shape); dt_num=np.zeros(shape); dt_ratio=np.zeros(shape)
dt_diff = np.zeros(shape);
eta_AD_arr = np.zeros(shape)   # for the closure check

for i in range(T.size):
    print("visc:",i)
    for j in range(B.size):
        for k in range(rho.size):
            HII=get_HII(logT[i])
            ne=get_ne(logT[i],rho[k],HII)
            nN=get_nN(rho[k],HII)
            Lam=get_Lambda(T[i],ne)
            ni=ne/1.2
            bi=get_omega_ci(B[j])/(get_nu_ii(T[i],ni,Lam)+get_nu_in(T[i],nN))
            nu_ion=get_nu_ion(T[i],rho[k],Lam)

            Z  = get_zeta_over_rho(T[i],logT[i],rho[k])
            X  = HII
            Vn = get_nu_neu(T[i],rho[k])
            VP = nu_ion
            Vp1= brag_perp1(nu_ion,bi)
            Vp2= brag_perp2(nu_ion,bi)
            Vw1= brag_wedge1(nu_ion,bi)
            Vw2= brag_wedge2(nu_ion,bi)

            rho_n=(1.0-HII)*rho[k]; rho_i=HII*rho[k]; v_in=get_nu_in(T[i],nN)   # for the closure check
            eta_AD_arr[i,j,k] = B[j]**2*rho_n**2/(4*np.pi*rho_i*v_in*rho[k]**2) if rho_n*v_in > 0.0 else 0.0

            # ---- analytic bound ----
            lA=lam_A_max(Z,X,Vn,VP,Vp1,Vp2); lC=lam_C_max(X,Vw1,Vw2)
            dt_pp = dx*dx/(2.0*lA) if lA>0 else np.inf
            dt_wg = dx*dx/((4.0/np.sqrt(3.0))*np.abs(lC)) if np.abs(lC)>0 else np.inf
            dt_bound[i,j,k]=1.0/(1.0/dt_pp + 1.0/dt_wg)

            # ---- true RK3 limit from the numerical spectrum ----
            lams=[]
            for b_ in ([(np.sqrt(u_), np.sqrt(1.0-u_), 0.0) for u_ in U_SCAN] if SCAN_ORIENTATION else [(BX,BY,BZ)]):
                Mv=build_visc(b_,Z,X,Vn,VP,Vp1,Vp2,Vw1,Vw2)
                lams.append(np.linalg.eigvals(diff*Mv))
            dt_num[i,j,k]=rk3_cfl(np.concatenate(lams), R_rk3)
            
            dt_diff[i,j,k] = dt_num[i,j,k] - dt_bound[i,j,k]
            dt_ratio[i,j,k]=dt_bound[i,j,k]/dt_num[i,j,k]

print("orientation:", f"scan over u = cos^2(theta), {U_SCAN.size} values" if SCAN_ORIENTATION else f"b = ({BX}, {BY}, {BZ})", "at psi = pi/2")
print("\nfinite dt_num:", np.all(np.isfinite(dt_num)))
print("bound violations (dt_bound > dt_num):", int(np.sum(dt_ratio>1.0+1e-12)))   # relative tolerance: ties to round-off are not violations
print("most conservative ratio dt_bound/dt_num:", np.nanmin(dt_ratio))
print("least conservative ratio dt_bound/dt_num:", np.nanmax(dt_ratio))

#%%
for i in range(np.size(T)):
    for j in range(np.size(B)):
        for k in range(np.size(rho)):
            if not is_valid(T[i], B[j], rho[k], eta_AD_arr[i,j,k], dx_phys):
               dt_bound[i,j,k] = np.nan
               dt_num[i,j,k]   = np.nan
               dt_diff[i,j,k]  = np.nan
               dt_ratio[i,j,k]   = np.nan
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

def grid_maps(cube, title, cbar_label):
    a = np.abs(cube[np.isfinite(cube) & np.isfinite(cube)])
    a = a[a > 0]
    if a.max() < 10.0*a.min():                             # under a decade
        norm = mcolors.Normalize(vmin=a.min(), vmax=a.max())
    else:
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
          r'$\Delta t_{\rm bound}/\Delta t_{\rm num}$')