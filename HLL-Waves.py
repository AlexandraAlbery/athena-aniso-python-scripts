# This script derives the wavespeeds for HLLE-Mod

##-----------------------------------------------------------------------------
# SETUP
##-----------------------------------------------------------------------------

import sympy as sy

x, y, z = sy.symbols(r"x y z", real = True)
r = sy.Matrix(3,1, [x, y, z])
etaH, t = sy.symbols(r"\eta_{H} t", positive = True)
Eint = sy.Function(r"\varepsilon", positive=True)(x, t)

rho, E, P = [f(x,t) for f in sy.symbols(r"\rho E P", cls=sy.Function, positive=True)]
vx, vy, vz, Bx, By, Bz = [f(x,t) for f in sy.symbols(r"v_{x} v_{y} v_{z} B_{x} B_{y} B_{z}", cls=sy.Function, real=True)]
bx, by, bz = [f(x,t) for f in sy.symbols(r"b_{x} b_{y} b_{z}", cls=sy.Function, real=True)]
gamma = sy.symbols("gamma", positive = True)
v = sy.Matrix(3,1, [vx, vy, vz])
B = sy.Matrix(3,1, [Bx, By, Bz])
b = sy.Matrix(3,1, [bx, by, bz])
I = sy.eye(3)
theta, phi = sy.symbols(r"\theta \phi",real=True)


def curl_x(V):          # 1D: d_y = d_z = 0
    return sy.Matrix([0, -sy.diff(V[2], x), sy.diff(V[1], x)])
def grad_x(f):
    return sy.Matrix([sy.diff(f, x), 0, 0])
def div_x(f):   return sy.diff(f[0], x) if f.shape[1] == 1 else f[0, :].T.diff(x)
def div3(V):
    return sum(sy.diff(V[i], r[i]) for i in range(3))
def curl3(V):
    return sy.Matrix([sy.diff(V[2], y) - sy.diff(V[1], z),
                      sy.diff(V[0], z) - sy.diff(V[2], x),
                      sy.diff(V[1], x)  - sy.diff(V[0], y)])
def outer(a, b): return a*b.T 

def lin(e):
    e = e.subs(pert, simultaneous=True).doit()            # d/dx
    return sy.expand(sy.diff(sy.expand(e), ep).subs(ep, 0)) 
def denest(a, b):        # sqrt(a + sqrt(b)), when a^2 - b is a perfect square
    d = sy.sqrt(sy.factor(sy.expand(a**2 - b)))
    return sy.sqrt(sy.cancel((a + d)/2)) + sy.sqrt(sy.cancel((a - d)/2))

def to_u(e):
      """to (c_a, c_s, D, u).  cos->sqrt(u), sin->sqrt(1-u) handles every
      power at once; subbing cos(theta)**2 misses cos(theta)**4."""
      e = sy.expand(sy.expand_trig(sy.expand(e)))
      e = e.subs([(B0, ca*sy.sqrt(rho0)),
                  (E0, cs**2*rho0/(gamma*(gamma - 1)))], simultaneous=True)
      e = sy.expand(e.subs(dk, sy.sqrt(D*cs**2*rho0/(gamma*(gamma - 1)))/k))
      e = e.subs([(sy.cos(theta), sy.sqrt(u)),
                  (sy.sin(theta), sy.sqrt(1 - u))], simultaneous=True)
      return sy.simplify(sy.expand(e))

#%%
##-----------------------------------------------------------------------------
# HALL EFFECT (WHISTLER WAVE)
##-----------------------------------------------------------------------------

P    = (gamma - 1)*Eint
Bsq  = B.dot(B)
Etot = Eint + rho*v.dot(v)/2 + Bsq/2
EH   = etaH*curl_x(B).cross(b) 

drho_dt = - div_x(rho*v)
dv_dt = -div_x(rho*outer(v,v) - outer(B,B) + (P + sy.Rational(1,2)*B.dot(B))*I)
dE_dt = -div_x((Etot + P + Bsq/2)*v - B*(B.dot(v)) + EH.cross(B))
dB_dt = curl_x(v.cross(B) - etaH*curl_x(B).cross(b))

subs_bhat = [(bx, sy.cos(theta)), (by, sy.sin(theta)*sy.cos(phi)), (bz, sy.sin(theta)*sy.sin(phi))]

dU = sy.Matrix([drho_dt, dv_dt[0], dv_dt[1], dv_dt[2], dE_dt, dB_dt[0], dB_dt[1], dB_dt[2]]).subs(subs_bhat, simultaneous=True)
 
dx_rho = dU[0]
dx_v   = sy.Matrix(3, 1, [(dU[1+i] - v[i]*dx_rho)/rho for i in range(3)])
dx_B   = sy.Matrix(3, 1, [dU[5+i] for i in range(3)])
dx_eps = dU[4] - v.dot(v)/2*dx_rho - rho*v.dot(dx_v) - B.dot(dx_B)
 
LHS = sy.Matrix(8, 1, [rho.diff(t), vx.diff(t), vy.diff(t), vz.diff(t),
                       Eint.diff(t), Bx.diff(t), By.diff(t), Bz.diff(t)])
RHS = sy.Matrix(8, 1, [dx_rho, dx_v[0], dx_v[1], dx_v[2], dx_eps,
                       dx_B[0], dx_B[1], dx_B[2]])
 
# div B = 0 in a 1D sweep: drops the monopole terms the conservative form carries
RHS = sy.simplify(RHS.subs({sy.Derivative(Bx, x): 0}))

rho0, E0, B0 = sy.symbols(r"\rho_0 \varepsilon_0 B_0", positive=True)
amps = sy.symbols(r"\delta\rho \delta{v_x} \delta{v_y} \delta{v_z} "
                  r"\delta\varepsilon \delta{B_y} \delta{B_z}")
drho, dvx, dvy, dvz, deps, dBy, dBz = amps
ep, k, w = sy.symbols(r"\epsilon k \omega", positive=True)
Ew = sy.exp(sy.I*(k*x - w*t))
b0 = [sy.cos(theta), sy.sin(theta)*sy.cos(phi), sy.sin(theta)*sy.sin(phi)]

pert = [(rho, rho0 + ep*drho*Ew), (Eint, E0 + ep*deps*Ew),   
        (vx, ep*dvx*Ew), (vy, ep*dvy*Ew), (vz, ep*dvz*Ew),
        (Bx, B0*b0[0]), # delta B_x = 0
        (By, B0*b0[1] + ep*dBy*Ew), (Bz, B0*b0[2] + ep*dBz*Ew)]

keep = [0, 1, 2, 3, 4, 6, 7] # B_x row is  0
M = sy.Matrix(7, 7, lambda i, j:
              sy.simplify(sy.expand(lin(RHS[keep[i]])/Ew).coeff(amps[j])))
    
lam = sy.symbols(r"\lambda")

cp = M.charpoly(lam).as_expr()

cp = sy.collect(sy.simplify(rho0**3*cp/(lam)),lam) # Cancel denominator of cp=0

omega = sy.symbols(r"\Omega")
ca, cs, u = sy.symbols("c_a c_s u", positive=True)
phys = [(B0, ca*sy.sqrt(rho0)), (E0, cs**2*rho0/(gamma*(gamma-1)))] # sound and alfven speeds

e   = cp.subs(lam**2, -omega*k**2) # sextic in lambda, even powers, cubic in omega
e   = e.subs(phys, simultaneous=True)
e   = e.subs(sy.cos(theta), sy.sqrt(u))       # cos^2 and cos^4
cp = sy.Poly(sy.expand(sy.simplify(e)), omega)
cp = sy.Poly(sy.expand(cp.as_expr()/cp.all_coeffs()[0]), omega)   # monic
cp = cp.as_expr()

domega_du = -sy.simplify(sy.diff(cp,u)/sy.diff(cp,omega))
u_star = sy.solve(domega_du,u)
cp_star = sy.simplify(cp.subs(u,list(u_star)[0]))

d2F_du2 = sy.simplify(sy.diff(cp,u,2))

sig = sy.solve(cp,omega)

roots_u0 = sy.roots(sy.Poly(cp.subs(u, 0), omega))
roots_u1 = sy.roots(sy.Poly(cp.subs(u, 1), omega))

A = 2*ca**2 + etaH**2*k**2 # Take square root of c2 and denest it gives whistler wave
c_w = sy.simplify(denest(A, sy.expand(A**2 - 4*ca**4))/sy.sqrt(2))

# checks against paper
Eh = etaH**2*k**2   # E in the paper
F_paper = -ca**4*cs**2*u**2 + u*(-omega**2*(Eh + ca**2) + omega*(Eh*cs**2 + ca**4 + 2*ca**2*cs**2)) + omega**2*(omega - ca**2 - cs**2)
print(sy.expand(cp - F_paper))                                                      # eq:F-hall
print(sy.expand(sy.diff(cp, u) - (-omega**2*(ca**2 + Eh) + omega*(ca**4 + 2*ca**2*cs**2 + Eh*cs**2) - 2*ca**4*cs**2*u)))  # dF/du
print(sy.expand(cp - ((1-u)*cp.subs(u, 0) + u*cp.subs(u, 1) + ca**4*cs**2*u*(1-u))))  # eq:chord

#%%
##-----------------------------------------------------------------------------
# PELTIER/SEEBECK (TE WAVE)
##-----------------------------------------------------------------------------

x, t = sy.symbols("x t", real=True)
kpar, kperp, dk = sy.symbols(r"\kappa_{J\parallel} \kappa_{J\perp} \Delta\kappa_{J}", positive=True)
theta, phi = sy.symbols(r"\theta \phi", real=True)

rho  = sy.Function(r"\rho", positive=True)(x, t)
Eint = sy.Function(r"\varepsilon", positive=True)(x, t)
vx, vy, vz, Bx, By, Bz = [f(x, t) for f in
    sy.symbols(r"v_x v_y v_z B_x B_y B_z", cls=sy.Function, real=True)]
v = sy.Matrix(3,1,[vx,vy,vz]); B = sy.Matrix(3,1,[Bx,By,Bz])
b = sy.Matrix(3,1,[sy.cos(theta), sy.sin(theta)*sy.cos(phi), sy.sin(theta)*sy.sin(phi)])
I = sy.eye(3)


P, Bsq = (gamma-1)*Eint, B.dot(B)
Etot = Eint + rho*v.dot(v)/2 + Bsq/2
Th   = P/rho                                        # Theta = P/rho
J    = curl_x(B)
bJ   = b.dot(J)
bG   = b.dot(grad_x(Th))
qJ   = dk*bJ*b              # Peltier heat flux, Delta kappa_J = x_ion(kappa_J,perp - kappa_J,par)
FT   = (1/Th)*(dk*bG*b)    # Seebeck EMF

drho_dt = - div_x(rho*v)
dv_dt = -div_x(rho*outer(v,v) - outer(B,B) + (P + sy.Rational(1,2)*B.dot(B))*I)
dE_dt = -div_x((Etot + P + Bsq/2)*v - B*(B.dot(v)) + qJ + FT.cross(B))
dB_dt = curl_x(v.cross(B) - FT)

dU = sy.Matrix(8,1, [drho_dt, dv_dt[0], dv_dt[1], dv_dt[2], dE_dt, dB_dt[0], dB_dt[1], dB_dt[2]])

# cons to prims
d_rho = dU[0]
d_v   = sy.Matrix(3,1,[(dU[1+i] - v[i]*d_rho)/rho for i in range(3)])
d_B   = sy.Matrix(3,1,[dU[5+i] for i in range(3)])
d_eps = dU[4] - v.dot(v)/2*d_rho - rho*v.dot(d_v) - B.dot(d_B)
RHS = sy.Matrix(8,1,[d_rho,d_v[0],d_v[1],d_v[2],d_eps,d_B[0],d_B[1],d_B[2]]) # dB/dx = 0
RHS = RHS.subs({sy.Derivative(Bx,x): 0})

ep,k,w = sy.symbols(r"\epsilon k \omega", positive=True)
rho0,E0,B0 = sy.symbols(r"\rho_0 \varepsilon_0 B_0", positive=True)
amps = sy.symbols(r"\delta\rho \delta{v_x} \delta{v_y} \delta{v_z} "
                  r"\delta\varepsilon \delta{B_y} \delta{B_z}")
Ew = sy.exp(sy.I*(k*x - w*t))
pert = [(rho, rho0+ep*amps[0]*Ew), (Eint, E0+ep*amps[4]*Ew),
        (vx, ep*amps[1]*Ew), (vy, ep*amps[2]*Ew), (vz, ep*amps[3]*Ew),
        (Bx, B0*b[0]), (By, B0*b[1]+ep*amps[5]*Ew), (Bz, B0*b[2]+ep*amps[6]*Ew)]
keep = [0,1,2,3,4,6,7]
M = sy.Matrix(7,7, lambda i,j: sy.simplify(sy.expand(lin(RHS[keep[i]])/Ew).coeff(amps[j])))
ca, cs, u = sy.symbols("c_a c_s u", positive=True)
D = sy.symbols("D", positive=True)
lam = sy.symbols(r"\lambda")
omega = sy.symbols(r"\Omega")
phys = [(B0, ca*sy.sqrt(rho0)), (E0, cs**2*rho0/(gamma*(gamma-1)))] # sound and alfven speeds

cp = M.charpoly(lam).as_expr()

# tidy cp = 0
cp = sy.simplify(cp*rho0/lam)
cp = sy.collect(cp,lam)
cp = sy.simplify(cp*(rho0**2)*E0)
cp = cp.subs(lam**2, -omega*k**2)
cp = to_u(cp)
cp = sy.Poly(sy.expand(cp), omega)
cp = sy.cancel(cp.as_expr()/cp.all_coeffs()[0])
cp = sy.collect(cp,omega)

c_iso = sy.symbols(r"c_{iso}", positive=True)

cp = sy.collect(sy.expand(cp).subs(cs**2, c_iso**2*gamma), omega)
cp = sy.collect(sy.expand(cp.subs(cs, sy.sqrt(gamma)*c_iso)), omega)

F = cp
dF = F.diff(omega)
d2F = F.diff(omega,2)
d3F = F.diff(omega,3)

# Paper check

cs2 = gamma*c_iso**2; cf2 = ca**2 + cs2
Lam = cf2 + D*u*(1-u)
F_paper = (omega - ca**2*u)*(omega**2 - cf2*omega + ca**2*cs2*u) - D*u*(1-u)*(omega**2 - (ca**2 + c_iso**2)*omega + ca**2*c_iso**2*u)
print(sy.simplify(F - F_paper))                                                          # eq:F-te
print(sy.simplify(F.subs(omega, Lam) - (D*u*(1-u)*Lam*(ca**2*(1-u) + c_iso**2)
  + ca**2*cs2*u*(cf2 - ca**2*u + D*u*(1-u)*(gamma-1)/gamma))))            # F(Lambda)
print(sy.simplify(dF.subs(omega, Lam) - (D**2*u**2*(1-u)**2 + D*ca**2*u*(3-2*u)*(1-u)
  + D*cs2*u*(1-u)*(2*gamma+1)/gamma + ca**4*(1-u) + cs2*(2*ca**2 + cs2))))  # F'(Lambda)
print(sy.simplify(d2F.subs(omega, Lam) - 2*(2*D*u*(1-u) + 2*ca**2 + 2*cs2 - ca**2*u)))   # F''(Lambda)
S = sy.zeros(7, 7)                                     # eq:S
S[0, 0] = gamma*E0/rho0**2; S[0, 4] = S[4, 0] = -1/rho0; S[4, 4] = 1/E0
for i in (1, 2, 3): S[i, i] = rho0
for i in (5, 6): S[i, i] = 1
print(sy.simplify(S*M + M.H*S) == sy.zeros(7, 7))  # True: S^(1/2) M S^(-1/2) skew-Hermitian

#%%
##-----------------------------------------------------------------------------
# GYRO-VISCOUS WAVE
##-----------------------------------------------------------------------------

x, t = sy.symbols("x t", real=True)
ca, cs, ciso, u = sy.symbols(r"c_a c_s c_{iso} u", positive=True)
Xi = sy.symbols(r"X", positive=True)
kpar, kperp, dk = sy.symbols(r"\kappa_{J\parallel} \kappa_{J\perp} \Delta\kappa_{J}", positive=True)
theta, phi = sy.symbols(r"\theta \phi", real=True)
rho0, E0, B0 = sy.symbols(r"\rho_0 \varepsilon_0 B_0", positive=True)
rho  = sy.Function(r"\rho", positive=True)(x, t)
Eint = sy.Function(r"\varepsilon", positive=True)(x, t)
vx, vy, vz, Bx, By, Bz = [f(x, t) for f in
    sy.symbols(r"v_x v_y v_z B_x B_y B_z", cls=sy.Function, real=True)]
v = sy.Matrix(3,1,[vx,vy,vz]); B = sy.Matrix(3,1,[Bx,By,Bz])
b = sy.Matrix(3,1,[sy.cos(theta), sy.sin(theta)*sy.cos(phi), sy.sin(theta)*sy.sin(phi)])
I = sy.eye(3)
Vw1, Vw2 = sy.symbols(r"\nu_{\wedge1} \nu_{\wedge2}", positive = True)

P, Bsq = (gamma-1)*Eint, B.dot(B)
Etot = Eint + rho*v.dot(v)/2 + Bsq/2
# --- projectors from the LOCAL bhat (NOT the section-0 bx,by,bz versions) ---
Bm = b*b.T
Ip = I - Bm
bc = sy.Matrix(3,3,[0,-b[2],b[1],b[2],0,-b[0], -b[1],b[0],0]) # cross product matrix for b

# --- (grad v)_ij = d_i v_j, only d_x nonzero; trace term uses div v ---
gradV = sy.Matrix(3,3,[sy.diff(v[0],x), sy.diff(v[1],x), sy.diff(v[2],x),
                       0,0,0, 0,0,0])
W  = gradV + gradV.T - sy.Rational(2,3)*div3(v)*sy.eye(3)
Pi = rho0*(-sy.Rational(1,2)*Vw1*Xi*(bc*W*Ip - Ip*W*bc)
           - 2*Vw2*Xi*(bc*W*Bm - Bm*W*bc))

drho_dt = - div_x(rho*v)
dv_dt = -div_x(rho*outer(v,v) - outer(B,B) + (P + sy.Rational(1,2)*B.dot(B))*I - Pi)
dE_dt = -div_x((Etot + P + Bsq/2)*v - B*(B.dot(v)) + Pi*v)
dB_dt = curl_x(v.cross(B))

dU = sy.Matrix(8,1, [drho_dt, dv_dt[0], dv_dt[1], dv_dt[2], dE_dt, dB_dt[0], dB_dt[1], dB_dt[2]])

# cons to prims
d_rho = dU[0]
d_v   = sy.Matrix(3,1,[(dU[1+i] - v[i]*d_rho)/rho for i in range(3)])
d_B   = sy.Matrix(3,1,[dU[5+i] for i in range(3)])
d_eps = dU[4] - v.dot(v)/2*d_rho - rho*v.dot(d_v) - B.dot(d_B)
RHS = sy.Matrix(8,1,[d_rho,d_v[0],d_v[1],d_v[2],d_eps,d_B[0],d_B[1],d_B[2]]) # dB/dx = 0
RHS = RHS.subs({sy.Derivative(Bx,x): 0})

ep,k,w = sy.symbols(r"\epsilon k \omega", positive=True)
rho0,E0,B0 = sy.symbols(r"\rho_0 \varepsilon_0 B_0", positive=True)
amps = sy.symbols(r"\delta\rho \delta{v_x} \delta{v_y} \delta{v_z} "
                  r"\delta\varepsilon \delta{B_y} \delta{B_z}")
Ew = sy.exp(sy.I*(k*x - w*t))
pert = [(rho, rho0+ep*amps[0]*Ew), (Eint, E0+ep*amps[4]*Ew),
        (vx, ep*amps[1]*Ew), (vy, ep*amps[2]*Ew), (vz, ep*amps[3]*Ew),
        (Bx, B0*b[0]), (By, B0*b[1]+ep*amps[5]*Ew), (Bz, B0*b[2]+ep*amps[6]*Ew)]
keep = [0,1,2,3,4,6,7]
M = sy.Matrix(7,7, lambda i,j: sy.simplify(sy.expand(lin(RHS[keep[i]])/Ew).coeff(amps[j])))
lam, omega = sy.symbols(r"\lambda \Omega")
cp = M.charpoly(lam).as_expr()
cp = sy.cancel(cp/(lam*k**6))
cp = sy.simplify(cp.subs(lam**2,-omega*k**2)) # cubic in omega
cp = to_u(cp)
cp = sy.collect(cp, omega)
cp0 = cp.subs(u,0)
cp1 = cp.subs(u,1)
c_g0 = sy.solve(cp0,omega)
c_gv1 = sy.solve(cp1,omega)

G = Xi**2 *(Vw1**2 * (1-u)**2 + 4*Vw1*Vw2*u*(1-u) + 4*Vw2**2 * u)
dG = G.diff(u)
u_star = sy.solve(dG,u)
G_star = sy.simplify(G.subs(u,list(u_star)[0]))
G1 = sy.simplify(G.subs(u,1))
G0 = sy.simplify(G.subs(u,0))

S = sy.zeros(7, 7)
S[0, 0] = gamma*E0/rho0**2; S[0, 4] = S[4, 0] = -1/rho0; S[4, 4] = 1/E0
for i in (1, 2, 3): S[i, i] = rho0
for i in (5, 6): S[i, i] = 1
print(sy.simplify(S*M + M.H*S) == sy.zeros(7, 7))  # True: S^(1/2) M S^(-1/2) skew-Hermitian
#%%
##-----------------------------------------------------------------------------
# PELTIER/SEEBECK  + HALL (TE WAVE)
##-----------------------------------------------------------------------------
 
x, t = sy.symbols("x t", real=True)
gamma = sy.symbols(r"\gamma", positive=True)
theta, phi = sy.symbols(r"\theta \phi", real=True)
 
etaH = sy.symbols(r"\eta_{H}", positive=True)
kpar, kperp, dk = sy.symbols(
    r"\kappa_{J\parallel} \kappa_{J\perp} \Delta\kappa_{J}", positive=True)
 
rho = sy.Function(r"\rho", positive=True)(x, t)
Eint = sy.Function(r"\varepsilon", positive=True)(x, t)
vx, vy, vz, Bx, By, Bz = [f(x, t) for f in sy.symbols(
    r"v_x v_y v_z B_x B_y B_z", cls=sy.Function, real=True)]
cf = sy.symbols(r"c_{f}", real=True)
 
v = sy.Matrix(3, 1, [vx, vy, vz])
B = sy.Matrix(3, 1, [Bx, By, Bz])
bhat = sy.Matrix(3, 1, [sy.cos(theta),
                        sy.sin(theta) * sy.cos(phi),
                        sy.sin(theta) * sy.sin(phi)])
I = sy.eye(3)
 
P = (gamma - 1) * Eint
Bsq = B.dot(B)
Etot = Eint + rho * v.dot(v) / 2 + Bsq / 2
Theta = P / rho                       # Theta = P/rho, so Theta_0 = c_iso^2
 
Jcur = curl_x(B)
EH = etaH * Jcur.cross(bhat)             # Hall EMF
qJ = dk * bhat.dot(Jcur) * bhat          # Peltier heat flux
FT = (1 / Theta) * (dk * bhat.dot(grad_x(Theta)) * bhat) # Seebeck EMF
 
drho_dt = -div_x(rho * v)
dv_dt = -div_x(rho * outer(v, v) - outer(B, B)
               + (P + sy.Rational(1, 2) * Bsq) * I)
dE_dt = -div_x((Etot + P + Bsq / 2) * v - B * (B.dot(v))
               + qJ + (EH + FT).cross(B))
dB_dt = curl_x(v.cross(B) - EH - FT)
 
dU = sy.Matrix(8, 1, [drho_dt, dv_dt[0], dv_dt[1], dv_dt[2],
                      dE_dt, dB_dt[0], dB_dt[1], dB_dt[2]])
 
# Conservative -> primitive time derivatives
d_rho = dU[0]
d_v = sy.Matrix(3, 1, [(dU[1 + i] - v[i] * d_rho) / rho for i in range(3)])
d_B = sy.Matrix(3, 1, [dU[5 + i] for i in range(3)])
d_eps = dU[4] - v.dot(v) / 2 * d_rho - rho * v.dot(d_v) - B.dot(d_B)
 
RHS = sy.Matrix(8, 1, [d_rho, d_v[0], d_v[1], d_v[2], d_eps,
                       d_B[0], d_B[1], d_B[2]])
# div.B = 0 in a 1D sweep: drops the monopole terms the conservative form carries
RHS = RHS.subs({sy.Derivative(Bx, x): 0})
 
ep, k, w = sy.symbols(r"\epsilon k \omega", positive=True)
rho0, E0, B0 = sy.symbols(r"\rho_0 \varepsilon_0 B_0", positive=True)
amps = sy.symbols(r"\delta\rho \delta{v_x} \delta{v_y} \delta{v_z} "
                  r"\delta\varepsilon \delta{B_y} \delta{B_z}")
 
Ew = sy.exp(sy.I * (k * x - w * t))
pert = [(rho, rho0 + ep * amps[0] * Ew), (Eint, E0 + ep * amps[4] * Ew),
        (vx, ep * amps[1] * Ew), (vy, ep * amps[2] * Ew), (vz, ep * amps[3] * Ew),
        (Bx, B0 * bhat[0]),                                   # delta B_x = 0
        (By, B0 * bhat[1] + ep * amps[5] * Ew),
        (Bz, B0 * bhat[2] + ep * amps[6] * Ew)]
 
keep = [0, 1, 2, 3, 4, 6, 7] # Bx is zero
M = sy.Matrix(7, 7, lambda i, j:
              sy.simplify(sy.expand(lin(RHS[keep[i]]) / Ew).coeff(amps[j])))
 
# ---------------------------------------------------------------------------
# Characteristic polynomial
# ---------------------------------------------------------------------------
 
ca, cs, ciso, u = sy.symbols(r"c_a c_s c_{iso} u", positive=True)
D, E = sy.symbols("D E", positive=True)
s = sy.symbols("s", real=True)          # s = -sgn(eta_H Delta kappa_J) = +-1
lam, omega = sy.symbols(r"\lambda \Omega")
 
cp = M.charpoly(lam).as_expr()
cp = sy.cancel(cp / lam)
 
cp = cp.subs(lam**2, -omega * k**2)
 
cp = sy.expand(sy.expand_trig(sy.expand(cp)))
cp = cp.subs([(B0, ca * sy.sqrt(rho0)),
              (E0, rho0 * ciso**2 / (gamma - 1))], simultaneous=True)
cp = cp.subs([(sy.cos(theta), sy.sqrt(u)),
              (sy.sin(theta), sy.sqrt(1 - u))], simultaneous=True)
cp = sy.expand(sy.simplify(cp))
cp = sy.collect(cp,omega)
 
F = cp.subs([(etaH, sy.sqrt(E) / k),
             (dk, -s * sy.sqrt(D) * ciso * sy.sqrt(rho0) / (k * sy.sqrt(gamma - 1)))],
            simultaneous=True)
F = sy.expand(sy.cancel(sy.expand(F) / (-k**6))).subs(s**2, 1)     # monic in omega
 
 
def Mop(a2, c2):
    """Magnetosonic quadratic at wave-normal parameter u."""
    return omega**2 - (a2 + c2) * omega + a2 * c2 * u
 
# Manual simplification of F
F_man = (omega-ca**2*u)*(omega**2 - cf**2 * omega +ca**2 * cs**2 * u) # ideal: adiabatic
F_man += - E*u*omega*(omega-cs**2) # Hall
F_man += -D*u*(1-u)*(omega**2 - (ca**2 + ciso**2)*omega + ca**2 * ciso**2 * u) # TE: isothermal
F_man += 2*s*ca*ciso*sy.sqrt((gamma-1)*D*E)*u*(1-u)*omega # Hall x TE
 
dF_man = sy.diff(F_man, omega)
d2F_man = sy.diff(dF_man, omega) 
d3F_man = sy.diff(d2F_man, omega)
 
Psi = sy.symbols(r"\Psi", positive = True)
 
c_tew0 = sy.sqrt(Psi**2 + sy.Rational(1,4)*D + sy.Rational(1,2)*sy.sqrt(D*E))
c_tew1 = sy.sqrt(cf**2 + sy.Rational(1,4)*D + sy.Rational(1,2)*sy.sqrt(D*E))
c_tew2 =sy.sqrt( (sy.Rational(1,2)*etaH*k  +sy.Rational(1,2)*sy.sqrt(etaH**2*k**2 + 4*ca**2))  **2 + sy.Rational(1,4)*D + sy.Rational(1,2)*sy.sqrt(D*E))
 
F_man_psi = F_man.subs(omega,c_tew0**2)
 
Ro = omega**2 - cf**2 * omega +ca**2 * cs**2 * u
Go = (omega- ca**2 * u)*Ro - E*u*omega*(omega-cs**2)
Bo = D*(Ro+ (gamma-1)*ciso**2 * (omega - ca**2*u)) - 2*s*ca*ciso*sy.sqrt((gamma-1)*D*E)*omega
 
def red(e):
    e = sy.expand(e).subs(cf**2, ca**2 + cs**2)
    e = sy.expand(e).subs(cs**2, gamma*ciso**2)
    return sy.expand(e)
 
print(sy.simplify(red(F_man - F)))     # manual form == charpoly of M
print(red(Go - u*(1-u)*Bo - F_man))      # decomposition
 
# ---------------------------------------------------------------------------
# F(c_TEw^2, u) >= 0: worst case, then expansion in powers of sqrt(D)
# ---------------------------------------------------------------------------
sq = sy.sqrt
Psi2 = Psi**2
Fhat = Go - u*(1-u)*(D*omega*(omega - ca**2) + 2*ca*cs*sq(D*E)*omega)     # s = -1, gamma -> inf
print(sy.simplify(red(F_man - Fhat - u*(1-u)*(D*ciso**2*(omega - ca**2*u)
      + 2*ca*sq(D*E)*omega*(cs + s*ciso*sq(gamma - 1))))))                 # F - Fhat
g0 = Go.subs(omega, Psi2)
g1 = sy.diff(Go, omega).subs(omega, Psi2)
g2 = sy.diff(Go, omega, 2).subs(omega, Psi2)/2
A = [g0,
     sq(E)/2*(g1 - 4*ca*cs*Psi2*u*(1-u)),
     (g1 + E*g2 - 4*u*(1-u)*(Psi2*(Psi2 - ca**2) + ca*cs*E))/4,
     sq(E)/8*(2*g2 + E - 4*u*(1-u)*(2*Psi2 - ca**2 + ca*cs)),
     (g2 + 3*E - 4*u*(1-u)*(2*Psi2 - ca**2 + E))/16,
     sq(E)/32*(3 - 8*u*(1-u)),
     (1 - 2*u)**2/64]
print(sy.simplify(red(Fhat.subs(omega, c_tew0**2) - sum(A[n]*sq(D)**n for n in range(7)))))

#%%
##-----------------------------------------------------------------------------
# SUBAPPENDIX C.2: CHECK EVERY STEP OF THE TEXT
##-----------------------------------------------------------------------------
# Follows the subsection equation by equation. Each inequality L >= R is checked as an exact
# identity L - R = C, with C written only in quantities that are non-negative

def check_C2():
    """Check every displayed step of Subappendix C.2: one PASS/FAIL line each."""
    ub = sy.symbols(r"\bar{u}", nonnegative=True)                                  # 1 - u
    df, dE, dw = sy.symbols(r"\delta_f \delta_E \delta_w", nonnegative=True)       # non-negatives
    gm1 = sy.symbols(r"\gamma_1", positive=True)                                   # gamma - 1 > 0
    gaps = {df: Psi2 - ca**2 - cs**2, dE: Psi2 - ca**2 - E, dw: (Psi2 - ca**2)**2 - E*Psi2, ub: 1 - u}
    base = {cf: sq(ca**2 + cs**2), ciso: cs/sq(1 + gm1), gamma: 1 + gm1}        # c_s^2 = gamma c_iso^2
    fails = []

    def report(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n      {detail}" if detail and not ok else ""))
        if not ok:
            fails.append(name)

    def residual(e, extra):
        e = e.subs(extra) if extra else e
        r = sy.expand(sy.expand(e).subs(gaps).subs(base))
        return r if r == 0 else sy.simplify(sy.radsimp(r))

    def identity(name, lhs, rhs, extra=None):
        """lhs = rhs exactly."""
        r = residual(lhs - rhs, extra)
        report(name, r == 0, f"residual: {r}")

    def claim(name, lhs, rhs, cert, extra=None):
        """lhs >= rhs: lhs - rhs = cert exactly, and cert >= 0 term by term."""
        r = residual(lhs - rhs - cert, extra)
        report(name, r == 0 and cert.is_nonnegative is True,
               f"identity residual: {sy.factor(r)}" if r != 0 else "certificate not manifestly >= 0")

    Om = c_tew0**2                                       # c_TEw^2 = Psi^2 + D/4 + sqrt(DE)/2
    G = lambda W, uu=u: Go.subs({omega: W, u: uu})

    # ---------------------------------------------------------------------------
    print("--- Three non-negative gaps, Eq. (psi) ---")
    rE = sq(E*(E + 4*ca**2))
    cw2, cw2m = ca**2 + E/2 + rE/2, ca**2 + E/2 - rE/2      # the two roots of (Omega - c_a^2)^2 - E Omega
    tf, tw = sy.symbols(r"\tau_f \tau_w", nonnegative=True)  # Psi^2 - c_f^2, Psi^2 - c_w^2 >= 0
    onF, onW = {Psi: sq(cf**2 + tf)}, {Psi: sq(cw2 + tw)}
    quad = lambda W: (W - ca**2)**2 - E*W
    claim("delta_f >= 0  (Psi >= c_f)", Psi2 - cf**2, 0, tf, onF)
    identity("c_w^2 = c_a^2 + E/2 + sqrt(E(E+4c_a^2))/2 is c_{eta_H,4}^2 of Eq. (Hall-bound), E = eta_H^2 k^2",
             ((etaH*k/2 + sq(etaH**2*k**2 + 4*ca**2)/2)**2).subs(etaH, sq(E)/k), cw2)
    identity("c_w^2 is a root of (Omega - c_a^2)^2 - E Omega", quad(cw2), 0)
    identity("    and so is c_a^2 + E/2 - sqrt(E(E+4c_a^2))/2", quad(cw2m), 0)
    claim("    c_w^2 is the larger root", cw2, cw2m, rE)
    claim("sqrt(E(E+4c_a^2)) >= E, so c_w^2 >= c_a^2 + E", cw2, ca**2 + E, 2*ca**2*sq(E)/(sq(E + 4*ca**2) + sq(E)))
    claim("delta_E >= 0  (Psi^2 >= c_w^2)", Psi2 - ca**2 - E, 0, tw + 2*ca**2*sq(E)/(sq(E + 4*ca**2) + sq(E)), onW)
    claim("delta_w >= 0  (the quadratic is >= 0 above its larger root)", (Psi2 - ca**2)**2 - E*Psi2, 0,
          tw**2 + tw*rE, onW)
    identity("Psi^2 - c_a^2 = delta_f + c_s^2", Psi2 - ca**2, df + cs**2)
    identity("Psi^2 - c_s^2 = delta_f + c_a^2", Psi2 - cs**2, df + ca**2)
    identity("Psi^2 - 2 c_a c_s = delta_f + (c_a - c_s)^2", Psi2 - 2*ca*cs, df + (ca - cs)**2)

    # ---------------------------------------------------------------------------
    print("--- Step 1: reduction to a sign condition ---")
    Fp = sy.Poly(sy.expand(F_man), omega)
    report("F is a monic cubic in Omega", Fp.degree() == 3 and Fp.LC() == 1)
    A2 = Fp.coeff_monomial(omega**2)
    identity("A_2 = -[c_f^2 + (c_a^2 + E)u + D u(1-u)]", A2, -(cf**2 + (ca**2 + E)*u + D*u*(1 - u)))
    claim("2 c_TEw^2 + A_2 = delta_f + delta_E + (c_a^2+E)(1-u) + D[1 + (1-2u)^2]/4 + sqrt(DE) >= 0",
          2*Om + A2, 0, df + dE + (ca**2 + E)*ub + D*(1 + (1 - 2*u)**2)/4 + sq(D*E))

    # ---------------------------------------------------------------------------
    print("--- Step 2: the worst case over s and gamma ---")
    identity("s c_iso sqrt(gamma-1) = s c_s sqrt(1 - 1/gamma)", s*ciso*sq(gamma - 1), s*cs*sq(1 - 1/gamma))
    claim("    ... <= c_s", cs - ciso*sq(gamma - 1), 0, cs/(sq(1 + gm1)*(sq(1 + gm1) + sq(gm1))))
    claim("    ... >= -c_s", cs + ciso*sq(gamma - 1), 0, cs + cs*sq(gm1)/sq(1 + gm1))
    bracket = D*ciso**2*(omega - ca**2*u) + 2*ca*sq(D*E)*omega*(cs + s*ciso*sq(gamma - 1))
    identity("F - Fhat = u(1-u){D c_iso^2 (Omega - c_a^2 u) + 2 c_a sqrt(DE) Omega [c_s + s c_iso sqrt(gamma-1)]}",
             F_man - Fhat, u*(1 - u)*bracket)
    lim = sy.limit(bracket.subs({s: -1, ciso: cs/sq(gamma)}), gamma, sy.oo)
    report("    so Fhat is F at s = -1, gamma -> infinity with c_s fixed", sy.simplify(lim) == 0)
    bs = sy.symbols("b_s", nonnegative=True)            # c_s + s c_iso sqrt(gamma - 1), between 0 and 2 c_s
    dF = lambda sgn: {s: sgn, bs: cs + sgn*ciso*sq(gamma - 1)}
    cFF = u*ub*(D*ciso**2*(df + cs**2 + ca**2*ub + D/4 + sq(D*E)/2) + 2*ca*sq(D*E)*Om*bs)
    for sgn in (1, -1):
        claim(f"F - Fhat >= 0 at Omega = c_TEw^2, s = {sgn:+d}", (F_man - Fhat).subs(omega, Om), 0, cFF, dF(sgn))
    claim("c_TEw^2 >= Psi^2", Om, Psi2, D/4 + sq(D*E)/2)
    claim("Psi^2 >= c_a^2 u", Psi2, ca**2*u, df + cs**2 + ca**2*ub)

    # ---------------------------------------------------------------------------
    print("--- Step 3: expansion in sqrt(D) ---")
    h = D/4 + sq(D*E)/2
    g0t = G(Psi2)
    g1t = 3*Psi2**2 - 2*(cf**2 + (ca**2 + E)*u)*Psi2 + (ca**2*(cf**2 + cs**2) + E*cs**2)*u
    g2t = 3*Psi2 - cf**2 - (ca**2 + E)*u
    identity("g_1 = d_Omega G(Psi^2, u)", sy.diff(Go, omega).subs(omega, Psi2), g1t)
    identity("g_2 = d^2_Omega G(Psi^2, u)/2", sy.diff(Go, omega, 2).subs(omega, Psi2)/2, g2t)
    identity("G(Psi^2 + h, u) = g_0 + g_1 h + g_2 h^2 + h^3", G(Psi2 + h), g0t + g1t*h + g2t*h**2 + h**3)
    K2 = Psi2*(Psi2 - ca**2) + ca*cs*E
    a = [g0t,
         sq(E)/2*(g1t - 4*ca*cs*Psi2*u*(1 - u)),
         (g1t + E*g2t - 4*u*(1 - u)*K2)/4,
         sq(E)/8*(2*g2t + E - 4*u*(1 - u)*(2*Psi2 - ca**2 + ca*cs)),
         (g2t + 3*E - 4*u*(1 - u)*(2*Psi2 - ca**2 + E))/16,
         sq(E)/32*(3 - 8*u*(1 - u)),
         (1 - 2*u)**2/64]
    identity("Fhat(c_TEw^2, u) = sum_k a_k D^(k/2)", Fhat.subs(omega, Om), sum(a[n]*sq(D)**n for n in range(7)))
    report("    the a_k are those of your script's A list", all(residual(a[n] - A[n], None) == 0 for n in range(7)))

    # ---------------------------------------------------------------------------
    print("--- Step 4: every coefficient is non-negative ---")
    identity("u^2 coefficient of G is -c_a^4 c_s^2", sy.Poly(sy.expand(Go), u).coeff_monomial(u**2), -ca**4*cs**2)
    lin = lambda name, e: identity(f"{name} = (1-u) [{name}](0) + u [{name}](1)", e,
                                   (1 - u)*e.subs(u, 0) + u*e.subs(u, 1))
    lin("g_1", g1t); lin("g_1 + E g_2", g1t + E*g2t); lin("2 g_2 + E", 2*g2t + E)
    gap1 = dw + ca**2*(df + dE) + df*(cs**2 + df + dE)                  # g_1(1) - c_a^2 c_s^2
    identity("g_1(0) = Psi^4 + 2 Psi^2 delta_f", g1t.subs(u, 0), Psi2**2 + 2*Psi2*df)
    identity("g_1(1) = c_a^2 c_s^2 + delta_w + c_a^2(delta_f + delta_E) + delta_f(c_s^2 + delta_f + delta_E)",
             g1t.subs(u, 1), ca**2*cs**2 + gap1)
    identity("(g_1 + E g_2)(0) = Psi^2(Psi^2 + 2E) + delta_f(2 Psi^2 + E)", (g1t + E*g2t).subs(u, 0),
             Psi2*(Psi2 + 2*E) + df*(2*Psi2 + E))
    identity("(g_1 + E g_2)(1) = Psi^2(Psi^2 - c_a^2) + 2 delta_f(Psi^2 - c_a^2) + (c_a^2 + E) delta_E",
             (g1t + E*g2t).subs(u, 1), Psi2*(Psi2 - ca**2) + 2*df*(Psi2 - ca**2) + (ca**2 + E)*dE)
    identity("(2 g_2 + E)(0) = 4 Psi^2 + 2 delta_f + E", (2*g2t + E).subs(u, 0), 4*Psi2 + 2*df + E)
    identity("(2 g_2 + E)(1) = 2 Psi^2 + 3 delta_f + delta_E + c_s^2", (2*g2t + E).subs(u, 1),
             2*Psi2 + 3*df + dE + cs**2)
    lb1 = (1 - u)*Psi2**2 + u*ca**2*cs**2
    lb2 = (1 - u)*Psi2*(Psi2 + 2*E) + u*Psi2*(Psi2 - ca**2)
    lb3 = (1 - u)*4*Psi2 + 2*u*Psi2
    gap2 = ub*df*(2*Psi2 + E) + u*(2*df*(df + cs**2) + (ca**2 + E)*dE)
    gap3 = ub*(2*df + E) + u*(3*df + dE + cs**2)
    claim("g_1 >= (1-u) Psi^4 + u c_a^2 c_s^2", g1t, lb1, ub*2*Psi2*df + u*gap1)
    claim("g_1 + E g_2 >= (1-u) Psi^2(Psi^2 + 2E) + u Psi^2(Psi^2 - c_a^2)", g1t + E*g2t, lb2, gap2)
    claim("2 g_2 + E >= (1-u) 4 Psi^2 + 2u Psi^2", 2*g2t + E, lb3, gap3)

    # a_0
    identity("G(Omega, 0) = Omega^2 (Omega - c_f^2)", G(omega, 0), omega**2*(omega - cf**2))
    identity("G(Omega, 1) = (Omega - c_s^2)[(Omega - c_a^2)^2 - E Omega]", G(omega, 1),
             (omega - cs**2)*((omega - ca**2)**2 - E*omega))
    identity("Eq. (chord): G = (1-u) G(Omega,0) + u G(Omega,1) + c_a^4 c_s^2 u(1-u)", Go,
             (1 - u)*G(omega, 0) + u*G(omega, 1) + ca**4*cs**2*u*(1 - u))
    identity("Eq. (chord) at Psi^2: a_0 = (1-u) Psi^4 delta_f + u(Psi^2 - c_s^2) delta_w + c_a^4 c_s^2 u(1-u)",
             a[0], (1 - u)*Psi2**2*df + u*(Psi2 - cs**2)*dw + ca**4*cs**2*u*(1 - u))
    c0 = ub*Psi2**2*df + u*(df + ca**2)*dw + ca**4*cs**2*u*ub
    claim("    so a_0 >= 0", a[0], 0, c0)
    # a_1
    sq1 = ((1 - u)*Psi2 - u*ca*cs)**2 + u*(1 - u)*(Psi2 - ca*cs)**2
    claim("a_1 >= sqrt(E)/2 [(1-u) Psi^4 + u c_a^2 c_s^2 - 4 c_a c_s Psi^2 u(1-u)]", a[1],
          sq(E)/2*(lb1 - 4*ca*cs*Psi2*u*(1 - u)), sq(E)/2*(ub*2*Psi2*df + u*gap1))
    identity("    [...] = [(1-u) Psi^2 - u c_a c_s]^2 + u(1-u)(Psi^2 - c_a c_s)^2",
             lb1 - 4*ca*cs*Psi2*u*(1 - u), sq1)
    c1 = sq(E)/2*(ub*2*Psi2*df + u*gap1 + (ub*Psi2 - u*ca*cs)**2 + u*ub*(Psi2 - ca*cs)**2)
    claim("    so a_1 >= 0", a[1], 0, c1)
    # a_2
    sq2 = (1 - 2*u)**2*Psi2*(Psi2 - ca**2) + (1 - u)**2*Psi2*(ca**2 + 2*E) + u*(1 - u)*(ca**2*Psi2 + 2*E*(Psi2 - 2*ca*cs))
    claim("4 a_2 >= (1-u) Psi^2(Psi^2+2E) + u Psi^2(Psi^2-c_a^2) - 4u(1-u)[Psi^2(Psi^2-c_a^2) + c_a c_s E]",
          4*a[2], lb2 - 4*u*(1 - u)*K2, gap2)
    identity("    [...] = (1-2u)^2 Psi^2(Psi^2-c_a^2) + (1-u)^2 Psi^2(c_a^2+2E) + u(1-u)[c_a^2 Psi^2 + 2E(Psi^2-2c_a c_s)]",
             lb2 - 4*u*(1 - u)*K2, sq2)
    c2 = (gap2 + (1 - 2*u)**2*Psi2*(df + cs**2) + ub**2*Psi2*(ca**2 + 2*E)
          + u*ub*(ca**2*Psi2 + 2*E*(df + (ca - cs)**2)))/4
    claim("    so a_2 >= 0", a[2], 0, c2)
    # a_3
    claim("c_a c_s <= Psi^2/2", Psi2/2, ca*cs, (df + (ca - cs)**2)/2)
    claim("    so 2 Psi^2 - c_a^2 + c_a c_s <= 5/2 Psi^2", sy.Rational(5, 2)*Psi2,
          2*Psi2 - ca**2 + ca*cs, ca**2 + (df + (ca - cs)**2)/2)
    claim("a_3 >= sqrt(E)/8 [(1-u) 4 Psi^2 + 2u Psi^2 - 10u(1-u) Psi^2]", a[3],
          sq(E)/8*(lb3 - 10*u*(1 - u)*Psi2), sq(E)/8*(gap3 + 4*u*ub*(ca**2 + (df + (ca - cs)**2)/2)))
    identity("    [...] = 2 Psi^2 [(1-2u)^2 + (1-u)^2]", lb3 - 10*u*(1 - u)*Psi2, 2*Psi2*((1 - 2*u)**2 + (1 - u)**2))
    c3 = sq(E)/8*(gap3 + 4*u*ub*(ca**2 + (df + (ca - cs)**2)/2) + 2*Psi2*((1 - 2*u)**2 + ub**2))
    claim("    so a_3 >= 0", a[3], 0, c3)
    # a_4, a_5, a_6
    identity("16 a_4 = delta_f + c_a^2(1-u) + E(2-u) + (1-2u)^2 (Psi^2 + delta_f + c_s^2 + E)",
             16*a[4], df + ca**2*(1 - u) + E*(2 - u) + (1 - 2*u)**2*(Psi2 + df + cs**2 + E))
    c4 = (df + ca**2*ub + E*(1 + ub) + (1 - 2*u)**2*(Psi2 + df + cs**2 + E))/16
    claim("    so a_4 >= 0", a[4], 0, c4)
    c5 = sq(E)/32*(1 + 2*(1 - 2*u)**2)
    claim("a_5 = sqrt(E)/32 [1 + 2(1-2u)^2] >= 0", a[5], 0, c5)
    c6 = (1 - 2*u)**2/64
    claim("a_6 = (1-2u)^2/64 >= 0", a[6], 0, c6)

    # ---------------------------------------------------------------------------
    print("--- Conclusion: F(c_TEw^2, u) as one sum of non-negative terms ---")
    cFhat = sum(c*sq(D)**n for n, c in enumerate([c0, c1, c2, c3, c4, c5, c6]))
    claim("Fhat(c_TEw^2, u) >= 0", Fhat.subs(omega, Om), 0, cFhat)
    for sgn in (1, -1):
        claim(f"F(c_TEw^2, u) >= Fhat(c_TEw^2, u) >= 0, s = {sgn:+d}, any gamma > 1", F_man.subs(omega, Om), 0,
              cFhat + cFF, dF(sgn))

    return fails

check_C2()