# This script recovers the eigenvalues used for CFL conditions in Bragsinkii MHD

# ============================================================================
# 0 SET-UP
# ============================================================================
 
import sympy as sy
from IPython.display import display, Math, Latex

x = sy.symbols('x', real=True)
 
# unit field direction b-hat and its skew matrix
bx, by, bz = sy.symbols(r'b_x b_y b_z', real=True)
bhat = sy.Matrix([bx, by, bz])
bhat_skew = sy.Matrix(3, 3, [  0, -bz,  by,
                              bz,   0, -bx,
                             -by,  bx,   0])   # == bhat.hat()
X, rho = sy.symbols(r'x_{\text{ion}} \rho', positive=True)
dt, dx, dp, E_int = sy.symbols(r"\Delta_{t} \Delta_{x} \Delta_{\perp} \epsilon_{\text{int}}", positive = True)

I  = sy.eye(3)
Bm = bhat * bhat.T          # parallel projector  b b^T
Ip = I - Bm                 # perpendicular projector
 
def curl_x(V):          # 1D: d_y = d_z = 0
    return sy.Matrix([0, -sy.diff(V[2], x), sy.diff(V[1], x)])
def grad_x(f):
    return sy.Matrix([sy.diff(f, x), 0, 0])
def dot(a, b):
    return (a.T*b)[0]
def coeff1(expr, f):                         # coefficient of f'(x)
    return sy.simplify(expr.coeff(sy.diff(f, x, 1)))
def coeff2(expr, f):    # coefficient of f''(x)
    return sy.simplify(expr.coeff(sy.diff(f, x, 2)))
def div3(V):
    return sum(sy.diff(V[i], r[i]) for i in range(3))
def curl3(V):
    return sy.Matrix([sy.diff(V[2], y) - sy.diff(V[1], z),
                      sy.diff(V[0], z) - sy.diff(V[2], x),
                      sy.diff(V[1], x)  - sy.diff(V[0], y)])
def rho_skew(S):                      # spectral radius of a real skew matrix
    return sy.sqrt(sy.simplify(-sy.trace(S*S)/2))
 
# reduce even powers of (b_x, b_y, b_z) to u = b_x^2 = cos^2(theta).
bx2, by2, bz2, s, u = sy.symbols(r'bx2 by2 bz2 s u', positive=True)
 
def collect_u(e):
    e = sy.expand(e).subs([(bx**2, bx2), (by**2, by2), (bz**2, bz2)])  # even powers -> squares
    e = sy.expand(e.subs(by2, s - bz2))                 # group b_y^2 + b_z^2 = s
    return sy.expand(e.subs(s, 1 - bx2).subs(bx2, u))   # roll gone; |b|=1; u=cos^2
 
# b spherical form)
theta, phi = sy.symbols(r'\theta \phi', real=True)
subs_trig = [(bx, sy.cos(theta)), (by, sy.sin(theta)*sy.cos(phi)), (bz, sy.sin(theta)*sy.sin(phi))]
 
def to_sphere(M):
    Mt = M.subs(subs_trig, simultaneous=True)
    if hasattr(Mt, 'applyfunc'):
        return Mt.applyfunc(lambda e: sy.trigsimp(sy.expand_trig(sy.expand(e))))
    else:
        return sy.trigsimp(sy.expand_trig(sy.expand(Mt)))
 
# eigenvalue symbol shared by all characteristic polynomials
L = sy.symbols('L')
 
# --- output helpers ---------------------------------------------------------
def banner(title):
    print("\n" + "=" * 79 + "\n" + title + "\n" + "=" * 79)
 
def sqrt_factor(e):
    """Factor inside square roots (exposes perfect squares that
    simplify misses"""
    return e.replace(lambda a: a.is_Pow and a.exp == sy.Rational(1, 2),
                     lambda a: sy.sqrt(sy.factor(sy.expand(a.base))))
#%%
# ============================================================================
# 1. THERMAL CONDUCTION
# ============================================================================

banner("1a. THERMAL CONDUCTION (isotropic + parallel + perp)")
# --- parabolic conduction ----------------------

# --- parabolic conduction: q_aniso from eq. (kappa-aniso) ----------------------
k0, dkap, kw, gamma = sy.symbols(r'\kappa_{0} \Delta_{\kappa} \kappa_{w} \gamma', real=True)
kn, kpar, kperp, kwedge = sy.symbols(r'\kappa_{n} \kappa_{\parallel} \kappa_{\perp} \kappa_{\wedge}', real=True)
E_int_func = sy.Function(r'\epsilon_{\text{int}}')(x)
Theta_k = (gamma - 1)*E_int_func/rho              # P/rho, rho constant
gradTh  = grad_x(Theta_k)
bGk     = dot(bhat, gradTh)

q_neu   = -kn*rho*gradTh
q_par   = -kpar*rho*bGk*bhat
q_perp  = -kperp*rho*(gradTh - bGk*bhat)
q_wedge = -kwedge*rho*bhat_skew*gradTh
q_aniso = (1 - X)*q_neu + X*(q_par + q_perp + q_wedge)

dt_E_int_func = sy.expand(-sy.diff(q_aniso[0], x))   # d_t E_int_func = -div q_aniso
lam_k   = coeff2(dt_E_int_func, E_int_func)
lam_k_u = collect_u(lam_k)

display(Math((r"\lambda_{\kappa,\parallel\perp}(0) = " + sy.latex(sy.factor(lam_k_u.subs(u, 0))))))
display(Math((r"\lambda_{\kappa,\parallel\perp}(1) = " + sy.latex(sy.factor(lam_k_u.subs(u, 1))))))

# --- hyperbolic conduction: same q_wedge, b free in 3-D ------------------------
banner("1b. THERMAL CONDUCTION (wedge)")
y, z = sy.symbols('y z', real=True)
r = (x, y, z)
bfun = sy.Matrix([sy.Function('bx')(*r), sy.Function('by')(*r), sy.Function('bz')(*r)])
q_wedge_3d = -X*kwedge*rho*bfun.cross(sy.Matrix([sy.diff(Theta_k, x), 0, 0]))
adv_w = sy.expand(-div3(q_wedge_3d))

lam_kw = coeff1(adv_w, E_int_func)
kx = sy.symbols('k_x', real=True)
sym_c1 = sy.simplify(sy.expand_complex(      # centred-difference Fourier symbol
        (sy.exp(sy.I*kx*dx) - sy.exp(-sy.I*kx*dx))))   # -> I sin(k_x dx)/dx
lam_kw = lam_kw*sym_c1
display(Math(r"\lambda_{\kappa,\wedge} =" + sy.latex(sy.factor(lam_kw))))

#%% 
# ============================================================================
# 2a. VISCOSITY: bulk + isotropic + Braginskii para, perp1, perp2
# ============================================================================

banner("2a. VISCOSITY (bulk + isotropic + parallel + perp)")
 
ddvx, ddvy, ddvz = sy.symbols(r'ddvx ddvy ddvz', real=True)
Z, X, Vn, VP, Vp1, Vp2 = sy.symbols(r'\zeta x_{\text{ion}} \nu_{n} \nu_{\parallel} \nu_{\perp1} \nu_{\perp2}', positive=True)
 
## Build iso and bulk
grad = sy.Matrix(1, 3, [1, 0, 0])              # x-row of the divergence
gradV = sy.Matrix(3, 3, [ddvx, ddvy, ddvz, 0, 0, 0, 0, 0, 0])
gradVT = sy.transpose(gradV)
D = gradV + gradVT
W = D - sy.Rational(2, 3)*ddvx*I
 
Pi_b = Z*ddvx*I
Pi_n = (1 - X)*Vn*W
 
## Build aniso  (Bm, Ip come from section 0)
Wp = sy.simplify(Ip * W * Ip)
bWb = sy.transpose(bhat)*W*bhat
Bm_I3 = Bm - sy.Rational(1, 3)*I
 
IpWBm = Ip * W * Bm
BmWIp = Bm * W * Ip
 
Pi_P  = sy.Rational(3, 2)*X*VP*Bm_I3*bWb[0]
Pi_p1 = X*Vp1*(Wp + sy.Rational(1, 2)*Ip*bWb[0])
Pi_p2 = 4*X*Vp2*(IpWBm + BmWIp)
 
Pi_visc = Pi_b + Pi_n + Pi_P + Pi_p1 + Pi_p2
 
de_visc = grad*Pi_visc
 
dv = sy.Matrix([ddvx, ddvy, ddvz])
A = sy.Matrix([[sy.expand(e).coeff(v) for v in dv] for e in de_visc])
 
TrA  = collect_u(sy.trace(A))                          # Tr(A)
TrAA = sy.trace(A*A)
DA   = collect_u(sy.Rational(1, 2)*(TrA**2 - TrAA))    # 2nd invariant
DetA = collect_u(A.det())                              # det(A)
 
char_poly_A = sy.factor(-(L**3) + TrA*(L**2) - DA*L + DetA)
 
roots_A  = sy.solve(char_poly_A, L)
Lam_A    = [sy.simplify(r) for r in roots_A]
du_Lam_A = [sy.simplify(lam.diff(u)) for lam in Lam_A]
u_star_A = [sy.solve(dl, u) for dl in du_Lam_A]
    
## xy subsystem
subs_xy = [(bx, sy.sqrt(u)), (by, sy.sqrt(1 - u)), (bz, 0)]
A_u = A.subs(subs_xy, simultaneous=True).applyfunc(lambda e: sy.expand(sy.simplify(e)))
Axy   = A_u[:2, :2]
lam_z = sy.expand(A_u[2, 2])        # decoupled z eigenvalue
 
TrAxy  = sy.expand(sy.trace(Axy))
DetAxy = sy.expand(Axy.det())
 
print("char poly == -(L - lam_z)(L^2 - Tr(A_xy) L + Det(A_xy)):",
      sy.expand(char_poly_A + (L - lam_z)*(L**2 - TrAxy*L + DetAxy)) == 0)
 
lam_xy_p = (TrAxy + sy.sqrt(TrAxy**2 - 4*DetAxy))/2     # largest eigenvalue
lam_xy_m = (TrAxy - sy.sqrt(TrAxy**2 - 4*DetAxy))/2
 
display(Math(r"M_{\zeta,\nu, xy} = " + sy.latex(Axy)))
display(Math(r"\lambda_{z}(u) = " + sy.latex(lam_z)))
display(Math(r"\text{Tr}(M_{\zeta,\nu, xy})" + sy.latex(TrAxy)))
display(Math(r"Det(M_{\zeta,\nu, xy}) = " + sy.latex(sy.collect(DetAxy, X))))
 
dTrAxy  = sy.expand(sy.diff(TrAxy, u))
dDetAxy = sy.expand(sy.diff(DetAxy, u))
display(Math(r"d/du \text{Tr}(M_{\zeta,\nu, xy}) = " + sy.latex(dTrAxy)))
display(Math(r"d/du \text{Det}(M_{\zeta,\nu, xy}) = " + sy.latex(dDetAxy)))

u_star_A = sy.solve(sy.diff(lam_xy_p, u), u)
display(Math(r"u* = " + sy.latex([sy.simplify(us) for us in u_star_A])))

for lbl, uu in [("u=0", sy.S(0)), ("u=1", sy.S(1))] + \
               [("u*_%d" % (i + 1), us) for i, us in enumerate(u_star_A)]:
    display(Math(r"\lambda_{\zeta,\nu,xy}+(%s) = " % lbl + sy.latex(sy.simplify(sqrt_factor(lam_xy_p.subs(u, uu))))))

#%%
# ============================================================================
# 2b. GYROVISCOSITY wedge1 wedge2
# ============================================================================

banner("2b. GYROVISCOSITY")
 
Vw1, Vw2 = sy.symbols(r'\nu_{\wedge1} \nu_{\wedge2}', positive=True)
 
bc = bhat_skew          # (b x) cross-product matrix
 
WIp = W*Ip
IpW = Ip*W
bcWIp = bc*WIp
IpWbc = IpW*bc
 
Wb = W*Bm
bW = Bm*W
bcWb = bc*Wb
bWbc = bW*bc
 
Pi3 = -sy.Rational(1, 2)*Vw1*X*(bcWIp - IpWbc)
Pi4 = -2*Vw2*X*(bcWb - bWbc)
 
Pi_wedge = Pi3 + Pi4
 
de_wedge = grad*Pi_wedge
 
C = sy.Matrix([[sy.expand(e).coeff(v) for v in dv] for e in de_wedge])
 
TrC  = collect_u(sy.trace(C))
TrCC = sy.trace(C*C)
DC   = collect_u(sy.Rational(1, 2)*(TrC**2 - TrCC))
DetC = collect_u(C.det())
 
char_poly_C = sy.factor(-(L**3) + TrC*(L**2) - DC*L + DetC)
char_poly_C = sy.factor(char_poly_C, X)

Lam_test = X*sy.sqrt(-((Vw1*(1 - u) + 2*Vw2*u)**2) - 4*(Vw2**2)*u*(1 - u))
roots_C = [r for r in sy.solve(char_poly_C, L) if r != 0]
Lam_C = next((r for r in roots_C if sy.simplify(r - Lam_test) == 0), roots_C[-1])
 
d_Lam_C  = Lam_C.diff(u)
u_star_C = sy.solve(d_Lam_C, u)
 
crit   = Lam_C.subs(u, u_star_C[0])
point0 = Lam_C.subs(u, 0)
point1 = Lam_C.subs(u, 1)
 
display(Math(r"\lambda_{\nu\wedge}(u+) = " + sy.latex(Lam_C)))
display(Math(r"u^{*} = " + sy.latex(u_star_C)))
display(Math(r"\lambda_{\nu\wedge}(u*)" + sy.latex(sy.simplify(crit))))
display(Math(r"\lambda_{\nu\wedge}(u=0)" + sy.latex(point0)))
display(Math(r"\lambda_{\nu\wedge}(u=1)" + sy.latex(point1)))
 
A_sph  = to_sphere(A)
C_sph  = to_sphere(C)
AC_sph = A_sph + C_sph
print()
print("A symmetric?      ", sy.simplify(A_sph - A_sph.T) == sy.zeros(3, 3))
print("C skew-symmetric? ", sy.simplify(C_sph + C_sph.T) == sy.zeros(3, 3))
print("A+C symmetric?      ", sy.simplify(AC_sph - AC_sph.T) == sy.zeros(3, 3))
print("A+C skew-symmetric? ", sy.simplify(AC_sph + AC_sph.T) == sy.zeros(3, 3))
 
#%%
# ============================================================================
# 3. MAGNETIC DIFFUSION: OHMIC + AMBIPOLAR + HALL
# ============================================================================
banner("3. MAGNETIC DIFFUSION: Ohmic + ambipolar + Hall")
 
eta, etaAD, etaH = sy.symbols(r'\eta \eta_{AD} \eta_{H}', positive=True)

Bx = sy.Function('B_x')(x)
By = sy.Function('B_y')(x)
Bz = sy.Function('B_z')(x)
B  = sy.Matrix([Bx, By, Bz])
 
J     = curl_x(B)
Jpar  = dot(bhat, J)
Jperp = J - Jpar*bhat
 
ohmic = -eta*curl_x(curl_x(B))
ambi  = -etaAD*curl_x(Jperp)
hall  = -etaH*curl_x(bhat_skew*J)
OA = sy.expand(ohmic + ambi)
 
M_OA  = sy.Matrix(2, 2, [coeff2(OA[1], By),   coeff2(OA[1], Bz),
                         coeff2(OA[2], By),   coeff2(OA[2], Bz)])
M_H   = sy.Matrix(2, 2, [coeff2(hall[1], By), coeff2(hall[1], Bz),
                         coeff2(hall[2], By), coeff2(hall[2], Bz)])
M_OAH = M_OA + M_H
 
lam_OA = [collect_u(ev) for ev in M_OA.eigenvals()]
lam_H  = [ev for ev in M_H.eigenvals()]
 

lam_OAH = [collect_u(sy.simplify(ev)) for ev in M_OAH.eigenvals()]

lam_OAH_m, lam_OAH_p = lam_OAH
 
display(Math(r"M_{\eta,\parallel\perp}"+ sy.latex(M_OA)))
display(Math(r"M_{\eta,\wedge}"+ sy.latex(M_H)))
display(Math(r"\lambda_{\eta,\parallel\perp} = " + sy.latex(lam_OA)))
display(Math(r"\lambda_{\eta,\wedge} = " + sy.latex(lam_H)))
display(Math(r"\lambda_{\eta-} = " + sy.latex(lam_OAH_m)))
display(Math(r"\lambda_{\eta+} = " + sy.latex(lam_OAH_p)))
display(Math(r"\lambda_{\eta+}(0) = " + sy.latex(lam_OAH_m.subs(u,0))))
display(Math(r"\lambda_{\eta+}(1) = " + sy.latex(lam_OAH_p.subs(u,1))))
 
#%%
# ============================================================================
# 4. THERMOELECTRIC EFFECTS
# ============================================================================
banner("4. THERMOELECTRIC EFFECTS")
 
kJpar, kJperp, kJwedge = sy.symbols(r'\kappa_{J\parallel} \kappa_{J\perp} \kappa_{J\wedge}', real=True)
dkj, kwj = sy.symbols(r'\Delta_{J\kappa} \kappa_{Jw}', positive = True)    # E = frozen background epsilon_int
 
Theta   = (gamma - 1)*E_int_func/rho        # P/rho as a field
Theta0  = (gamma - 1)*E_int/rho              # frozen prefactor value
 
bJ = dot(bhat, J)
qJ = -X*(kJpar*bJ*bhat + kJperp*(J - bJ*bhat) + kJwedge*bhat_skew*J)
 
gradTheta = grad_x(Theta)
bG = dot(bhat, gradTheta)
FT = -(X/Theta0)*(kJpar*bG*bhat + kJperp*(gradTheta - bG*bhat)
                     + kJwedge*bhat_skew*gradTheta)
 
dt_eps = sy.expand(-sy.diff(qJ[0], x))   # energy:    d_t E_int = -div q_J
dt_B   = sy.expand(-curl_x(FT))          # induction: d_t B      = -curl F_T
 
# operator on (E_int_func, B_y, B_z)'' -- the E matrix of the paper
M_TE = sy.Matrix(3, 3, [coeff2(dt_eps,  E_int_func), coeff2(dt_eps,  By), coeff2(dt_eps,  Bz),
                            coeff2(dt_B[1], E_int_func), coeff2(dt_B[1], By), coeff2(dt_B[1], Bz),
                            coeff2(dt_B[2], E_int_func), coeff2(dt_B[2], By), coeff2(dt_B[2], Bz)])
 
#   Delta kappa_J = x_ion (kJperp - kJpar),   kappa_J,w = x_ion kJwedge
subs_TE = [(kJperp, kJpar + dkj/X), (kJwedge, kwj/X)]
M_TE = sy.expand(M_TE.subs(subs_TE))
display(Math(r"M_{TE} = "+ sy.latex(M_TE)))
 
# --- characteristic polynomial and eigenvalues -------------------------------
M_TE2 = M_TE*M_TE
 
TrTE  = sy.trace(M_TE)
DTE   = sy.simplify(sy.Rational(1, 2)*(TrTE**2 - sy.trace(M_TE2)))
DetTE = M_TE.det()
 
lam_TE_m = collect_u(sy.simplify(-sy.sqrt(sy.Rational(1, 2)*(sy.trace(M_TE2)))))
lam_TE_p = collect_u(sy.simplify(sy.sqrt(sy.Rational(1, 2)*(sy.trace(M_TE2)))))
 
# --- maximise over u ---------------------------------------------------------
dlam_TE_p = sy.diff(lam_TE_p, u)
dlam_TE_paper = (2*dkj**2*u - dkj**2 - kwj**2) \
    /(2*sy.sqrt(E_int)*sy.sqrt(dkj**2*u**2 - dkj**2*u - kwj**2*u + kwj**2))
 
u_star_p = sy.solve(dlam_TE_p, u)

lam_TE_max = []
for c in list(u_star_p) + [sy.S(0), sy.S(1)]:
    val = sy.simplify(lam_TE_p.subs(u, c))
    lam_TE_max.append(val)
    display(Math(r"u = " + sy.latex(sy.simplify(c)) + r":\quad \lambda_{TE,+} = " + sy.latex(val)))

#%%
# ============================================================================
# 5. COUPLED CONDUCTION + DIFFUSION + THERMOELECTRICS: WELL-POSEDNESS
# ============================================================================
banner("5. COUPLED SYSTEM (well-posedness criterion)")

psi = sy.symbols('psi', real=True)
cF  = -4*dt*sy.sin(psi)**2/dx**2         # common Fourier factor
 
D_T    = cF*(gamma - 1)*(k0 + dkap*sy.cos(theta)**2)
D_Tw   = 8*(gamma - 1)*sy.I*dt*kw*sy.sin(psi)*sy.cos(psi)/(dp*dx)
D_My   = cF*(eta + etaAD*(1 - sy.sin(theta)**2*sy.sin(phi)**2))
D_Mz   = cF*(eta + etaAD*(1 - sy.sin(theta)**2*sy.cos(phi)**2))
D_MAD  = cF*etaAD*sy.sin(theta)**2*sy.cos(phi)*sy.sin(phi)
D_H    = -cF*etaH*sy.cos(theta)
D_TEey = cF*(-dkj*sy.sin(phi)*sy.cos(theta)*sy.sin(theta) + kwj*sy.cos(phi)*sy.sin(theta))
D_TEez = cF*( dkj*sy.cos(phi)*sy.cos(theta)*sy.sin(theta) + kwj*sy.sin(phi)*sy.sin(theta))
D_TEBy = cF*( dkj*sy.cos(theta)*sy.sin(phi)*sy.sin(theta) + kwj*sy.cos(phi)*sy.sin(theta))/E_int
D_TEBz = cF*(-dkj*sy.cos(phi)*sy.cos(theta)*sy.sin(theta) + kwj*sy.sin(phi)*sy.sin(theta))/E_int
 
Fcpl = sy.Matrix(3, 3, [D_T + D_Tw, D_TEey,      D_TEez,
                        D_TEBy,     D_My,        D_MAD - D_H,
                        D_TEBz,     D_MAD + D_H, D_Mz])
display(Math("F = "+ sy.latex(Fcpl)))
 
# --- Bendixson bound after the similarity transform --------------------------
# F~ = P^-1 F P has the same spectrum as F; Bendixson on F~ bounds Re(lam(F))
# by the eigenvalues of H = (F~ + F~^dagger)/2.
P_sim = sy.diag(1, 1/sy.sqrt(E_int), 1/sy.sqrt(E_int))
Ft_cpl = P_sim.inv()*Fcpl*P_sim

display(Math(r"\tilde{F} = " + sy.latex(Ft_cpl)))

H_cpl = ((Ft_cpl + Ft_cpl.H)/2).applyfunc(lambda e: sy.simplify(sy.expand(e)))
H_cpl = sy.simplify(H_cpl/cF) # cancel off derivatives and sin^2(psi) for simplicity
display(Math(r"H_{\tilde{F}} = "+ sy.latex(H_cpl)))
 
Q_rot = sy.Matrix(3, 3, [1, 0, 0,
                         0, sy.cos(phi), sy.sin(phi),
                         0, sy.sin(phi), -sy.cos(phi)])
print("Q orthogonal:" + str(sy.simplify(Q_rot.T*Q_rot) == sy.eye(3)))

H_rot = (Q_rot.T*H_cpl*Q_rot).applyfunc(
            lambda e: sy.trigsimp(sy.expand_trig(sy.expand(e))))
display(Math(r"\phi\ \text{eliminated: } " + str(phi not in H_rot.free_symbols)))
display(Math(r"H_{\tilde{F},rot} = "+ sy.latex(H_rot)))

trH_rot  = sy.expand(H_rot[0, 0] + H_rot[1, 1])
detH_rot = sy.expand(H_rot[0, 0]*H_rot[1, 1] - H_rot[0, 1]**2)
lam_H_rot2 = (trH_rot  + sy.sqrt(trH_rot **2 - 4*detH_rot))/2
lam_H_rot3 = (trH_rot  - sy.sqrt(trH_rot **2 - 4*detH_rot))/2 # this branch can go negative

display(Math(sy.latex(lam_H_rot3)))

# --- the well-posed criterion, egienvalue must be positive -------------------

lam_HF_max = lam_H_rot3.subs(theta, sy.pi/2)
display(Math(sy.latex(lam_HF_max)))

#%%
# ============================================================================
# 6.a PARABOLIC COUPLED SYSTEM
# ============================================================================
 
banner("6a. PARABOLIC COUPLED SYSTEM (Thermal + Magnetic + Nernst/Ettinghausen")


D_T    = (gamma - 1)*(k0 + dkap*sy.cos(theta)**2)
D_My   = (eta + etaAD*(1 - sy.sin(theta)**2*sy.sin(phi)**2))
D_Mz   = (eta + etaAD*(1 - sy.sin(theta)**2*sy.cos(phi)**2))
D_MAD  = etaAD*sy.sin(theta)**2*sy.cos(phi)*sy.sin(phi)

D_TEey_w = (kwj*sy.cos(phi)*sy.sin(theta))
D_TEez_w = (kwj*sy.sin(phi)*sy.sin(theta))
D_TEBy_w = (kwj*sy.cos(phi)*sy.sin(theta))/E_int
D_TEBz_w = (kwj*sy.sin(phi)*sy.sin(theta))/E_int

M_TE_para = sy.Matrix(3,3, [D_T, D_TEey_w, D_TEez_w, 
                              D_TEBy_w, D_My, D_MAD,
                              D_TEBz_w, D_MAD, D_Mz])


M_TE_para_tilde = P_sim.inv()*M_TE_para*P_sim

M_TE_para_tilde_rot = Q_rot.T*M_TE_para_tilde*Q_rot

M_TE_para_tilde_rot = sy.simplify(sy.expand(M_TE_para_tilde_rot))

lam_TE_para = sy.simplify(M_TE_para_tilde_rot.eigenvals())

lam_TE_para_max = list(lam_TE_para)[2]

lam_TE_para_max_2 = sy.simplify(sy.expand(lam_TE_para_max*lam_TE_para_max)).subs(sy.cos(theta)**2,u).subs(sy.sin(theta)**2,1-u)

display(Math(r"\lambda_{TM,R}(\kappa = \kappa_{J} = 0)"+sy.latex(sy.simplify(list(lam_TE_para)[2].subs(dkap,0).subs(k0,0).subs(kwj,0))))) # only magnetic diffusion confirm
display(Math(r"\lambda_{TM,R}(\eta= \kappa_{J} = 0)"+sy.latex(sy.simplify(list(lam_TE_para)[2].subs(eta,0).subs(etaAD,0).subs(kwj,0))))) # only thermal diffusion confirm
display(Math(r"\lambda_{TM,R}(\kappa = \eta = 0)"+sy.latex(sy.simplify(list(lam_TE_para)[2].subs(eta,0).subs(etaAD,0).subs(k0,0).subs(dkap,0))))) # only TE confirm

display(Math(r"\lambda_{TM,R}(u=0)"+sy.latex(sy.simplify(list(lam_TE_para)[2].subs(theta,0))))) # u = 1 boundary
display(Math(r"\lambda_{TM,R}(u=1)"+sy.latex(sy.simplify(list(lam_TE_para)[2].subs(theta,sy.pi/2))))) # u = 0 boundary


#%%
# ============================================================================
# 6.b HYPERBOLIC COUPLED SYSTEM
# ============================================================================
 
banner("6b. HYPERBOLIC COUPLED SYSTEM (Hall + Peltier/Seebeck")

D_H    = -etaH*sy.cos(theta)
D_TEey_d = -dkj*sy.sin(phi)*sy.cos(theta)*sy.sin(theta) 
D_TEez_d =  dkj*sy.cos(phi)*sy.cos(theta)*sy.sin(theta)
D_TEBy_d =  dkj*sy.cos(theta)*sy.sin(phi)*sy.sin(theta)/E_int
D_TEBz_d = -dkj*sy.cos(phi)*sy.cos(theta)*sy.sin(theta)/E_int

M_TE_hyp = sy.Matrix(3,3, [0, D_TEey_d, D_TEez_d, 
                              D_TEBy_d, 0, -D_H,
                              D_TEBz_d, D_H, 0])

M_TE_hyp_tilde = P_sim.inv()*M_TE_hyp*P_sim

lam_TE_hyp = M_TE_hyp_tilde.eigenvals()

lam_TE_hyp_max = sy.Abs(list(M_TE_hyp_tilde.eigenvals())[1])

lam_TE_hyp_max = lam_TE_hyp_max.subs(sy.cos(theta),sy.sqrt(u)).subs(sy.sin(theta)**2,1-u)

d_lam_TE_hyp_max = lam_TE_hyp_max.diff(u)

u_star_TE_hyp = list(sy.solve(d_lam_TE_hyp_max,u))[0]

lam_TE_hyp_max_star = sy.simplify(lam_TE_hyp_max.subs(u,u_star_TE_hyp))

display(Math(r"u^{*} = "+sy.latex(u_star_TE_hyp)))
display(Math(r"\lambda_{TM,H}(0) = "+sy.latex(lam_TE_hyp_max.subs(u,0))))
display(Math(r"\lambda_{TM,H}(1) = "+sy.latex(lam_TE_hyp_max.subs(u,1))))
display(Math(r"\lambda_{TM,H}(u^{*}) = "+sy.latex(lam_TE_hyp_max_star)))

#%%
# ============================================================================
# 7 BIERMANN BATTERY (test)
# ============================================================================
banner("7. BIERMANN BATTERY (test)")

Bx = sy.Function(r'B_x')(x,y)
By = sy.Function(r'B_y')(x,y)
Bz = sy.Function(r'B_z')(x,y)
B = sy.Matrix([Bx,By,Bz])
rho = sy.Function(r'\rho')(x)
P = sy.Function(r'P')(y)
eta_BB = sy.Function(r"\eta_{BB}")(P,rho)

E_BB = -(eta_BB*(P/rho)*grad_x(sy.log(rho))+grad_x(eta_BB*(P/rho)))
S_BB = E_BB.cross(B)
dE_tot_dt = -div3(S_BB)
dB_dt = -curl3(E_BB)
dE_mag_dt = B.dot(dB_dt)              # d/dt (B^2/2) from the battery
dE_int_dt = dE_tot_dt - dE_mag_dt     # kinetic energy is untouched
J = curl3(B)
assert sy.simplify(dE_int_dt - J.dot(E_BB)) == 0
RHS = sy.Matrix([dE_int_dt,dB_dt[2]]) # dBx_dt = dBy_dt = 0
