# This script derives the analytical results for testing errors

import sympy as sy
from IPython.display import display, Math, Latex

def banner(title):
    print("\n" + "=" * 79 + "\n" + title + "\n" + "=" * 79)

x, y, z, t = sy.symbols('x y z t', real=True)
X = sy.Matrix([x, y, z])

def grad(f):
    return sy.Matrix([sy.diff(f, c) for c in X])


def div(F):
    return sum(sy.diff(F[i], X[i]) for i in range(3))


def curl(F):
    return sy.Matrix([sy.diff(F[2], y) - sy.diff(F[1], z),
                      sy.diff(F[0], z) - sy.diff(F[2], x),
                      sy.diff(F[1], x) - sy.diff(F[0], y)])


def div_tensor(P):          # (div P)_j = d_i P_ij
    return sy.Matrix([sum(sy.diff(P[i, j], X[i]) for i in range(3)) for j in range(3)])


def grad_vec(v):            # (grad v)_ij = d_i v_j
    return sy.Matrix(3, 3, lambda i, j: sy.diff(v[j], X[i]))


def viscous_stress(v, rho0, b, zeta, nu, xi, nu_par, nu_p1, nu_p2):
    """Pi = zeta div(v) I + (1-x) rho nu W + x [Pi_par + Pi(1) + Pi(2)], linearised (rho -> rho0)."""
    I3 = sy.eye(3)
    divv = div(v)
    W = grad_vec(v) + grad_vec(v).T - sy.Rational(2, 3)*divv*I3
    bb = b*b.T
    Ip = I3 - bb
    bWb = (b.T*W*b)[0]
    return (zeta*divv*I3 + (1 - xi)*rho0*nu*W
            + xi*(sy.Rational(3, 2)*rho0*nu_par*(bb - I3/3)*bWb
                  + rho0*nu_p1*(Ip*W*Ip + sy.Rational(1, 2)*Ip*bWb)
                  + 4*rho0*nu_p2*(Ip*W*bb + bb*W*Ip)))

# common symbols
K, rho0, P0, gam, B0, v0, xi = sy.symbols(r'K \rho_{0} P_{0} \gamma B_{0} v_{0} x_{\text{ion}}', positive=True)
kx, ky, kz = sy.symbols(r'k_{x} k_{y} k_{z}', real=True)

#%%
# =============================================================================== case 0
banner("0. THERMAL CONDUCTION (isotropic + parallel + perp)")
display(Math(r"T(x,0) = T_{0}\left(1 + A_{T}(0)\cos(k_{x}x)\right)"))
T0 = sy.symbols(r'T_{0}', positive=True)
kn, kpa, kpe = sy.symbols(r'\kappa_{n} \kappa_{\parallel} \kappa_{\perp}', positive=True)
bx, by, bz = sy.symbols("bx by bz", real=True)
b_hat = sy.Matrix(3,1,[bx,by,bz])
A = sy.Function(r'A_{T}')(t)
T = T0*(1 + A*sy.cos(kx*x + ky*y + kz*z))
gT = grad(T)
q = (gam-1)*((1-xi)*kn*gT + xi*(kpa*gT.dot(b_hat)*b_hat + kpe*(gT - gT.dot(b_hat)*b_hat)))
ode = sy.diff(T,t) - div(q)
A_sol = sy.dsolve(ode, A)
C1 = sy.symbols('C1')
A_sol = A_sol.subs(C1, A.subs(t, 0))

# decay rate D simplification
expo = A_sol.rhs.atoms(sy.exp).pop().args[0]
kp, kq = sy.symbols(r'k_{\parallel} k_{\perp}', positive=True) # wavenumbers
D_val = sy.expand(-expo/t)
D_val = sy.expand(D_val.subs(bx, (kp - by*ky - bz*kz)/kx))            # kx bx + ky by + kz bz = k_par
D_val = sy.expand(D_val.subs(kx**2, kp**2 + kq**2 - ky**2 - kz**2))   # kx^2 + ky^2 + kz^2 = k_par^2 + k_perp^2
D_val = sy.collect(sy.factor(D_val), [kp, kq])

D = sy.symbols(r'D', positive=True)
A_sol = A_sol.subs(sy.exp(expo), sy.exp(-D*t))
T_sol = T.subs(A, A_sol.rhs)
display(Math(sy.latex(A_sol)))
display(Math(r"D = " + sy.latex(D_val)))
display(Math(r"T(r,t) = " + sy.latex(T_sol)))

#%%
# =============================================================================== case 1
banner("1. VISCOSITY (bulk + isotropic + parallel + perp)")
theta = sy.symbols(r'\theta', real=True)                                   # b_tilt
K, B0, P0 = sy.symbols(r'K B_{0} P_{0}', positive=True)
zeta, nun, npa, np1, np2 = sy.symbols(r'\zeta \nu_{n} \nu_{\parallel} \nu_{\perp1} \nu_{\perp2}', positive=True)
a1, a2, a3 = sy.Function('a_1')(t), sy.Function('a_2')(t), sy.Function('a_3')(t)

r = sy.Matrix([x,y,z])
k = sy.Matrix([kx, 0, 0])
k_hat = sy.Matrix([1, 0, 0])                        # k = K x-hat
b_hat = sy.Matrix([0, sy.cos(theta), sy.sin(theta)])  # perpendicular to k
kxb = k_hat.cross(b_hat)                            # (0, -sin, cos)               
v = v0*sy.sin(r.dot(k))*(a1*k_hat + a2*b_hat + a3*kxb)
display(Math(r"v(r,0) = " + sy.latex(v.subs(t,0))))

I3 = sy.eye(3)
gv = sy.Matrix(3, 3, lambda i, j: sy.diff(v[j], (x, y, z)[i])) # (grad v)_ij = d_i v_j
divv = div(v)
W = gv + gv.T - sy.Rational(2, 3)*divv*I3
bb = b_hat*b_hat.T
Ip = I3 - bb
bWb = (b_hat.T*W*b_hat)[0]
Pi = (zeta*divv*I3 + (1-xi)*rho0*nun*W
      + xi*(sy.Rational(3, 2)*rho0*npa*(bb - I3/3)*bWb
            + rho0*np1*(Ip*W*Ip + Ip*bWb/2)
            + 4*rho0*np2*(Ip*W*bb + bb*W*Ip)))
divPi = sy.Matrix([sum(sy.diff(Pi[i, j], (x, y, z)[i]) for i in range(3)) for j in range(3)])

ode = sy.simplify(rho0*sy.diff(v,t) - divPi)
a_sol = sy.dsolve([ode[0], ode[1], ode[2]], [a1, a2, a3])
C1, C2, C3 = sy.symbols('C1 C2 C3')
a_sol = [s.subs({C1: a1.subs(t, 0), C2: a2.subs(t, 0), C3: a3.subs(t, 0)}) for s in a_sol]
tidy = lambda arg: sy.exp(-kx**2*t*sy.collect(sy.expand(-arg/(kx**2*t)), [zeta, nun], sy.factor))
a_sol = [sy.Eq(s.lhs, s.rhs.replace(sy.exp, tidy)) for s in a_sol]
display(list(a_sol)[0])
display(list(a_sol)[1])
display(list(a_sol)[2])

v_sol = v.subs({s.lhs: s.rhs for s in a_sol})
display(Math(r"v(r,t) = " + sy.latex(v_sol)))

#%%
# =============================================================================== case 2
banner("2. MAGNETIC DIFFUSION: Ohmic + ambipolar + Hall")
theta = sy.symbols(r'\theta', real=True)
eO, eA, eH, B0, dB0 = sy.symbols(r'\eta_{O} \eta_{A} \eta_{H} B_{0} \delta{B}_{0}', positive=True)
dBy, dBz = sy.Function(r'\delta B_y')(t), sy.Function(r'\delta B_z')(t)

b_hat = sy.Matrix([sy.cos(theta), sy.sin(theta), 0])          # B0 at theta to k = kx x-hat
B = B0*b_hat + sy.Matrix([0, dBy, dBz])*sy.cos(kx*x)          # background + transverse perturbation
J = curl(B)
E = eO*J + eH*J.cross(b_hat) + eA*(J - J.dot(b_hat)*b_hat)    # Ohmic + Hall + ambipolar
ode = sy.simplify(sy.diff(B, t) + curl(E))     # dB/dt = -curl E
display(Math(sy.latex(ode) + " = 0"))                         # x row is 0: B_x never changes

# 1) solve first: the two nontrivial rows for the two unknown amplitudes
B_sol = sy.dsolve([ode[1], ode[2]], [dBy, dBz], ics={dBy.subs(t, 0): 0, dBz.subs(t, 0): dB0}) # dBy(0) = 0, dBz(0) = dB_0
display(*B_sol)

# 2) substitutions afterwards: write the eta's through the physical rates
#    a_bar = eta_O + eta_A(1 + cos^2)/2,  delta_A = eta_A sin^2/2,  beta = eta_H cos,  D^2 = delta_A^2 - beta^2
a, d, D, beta = sy.symbols(r'\bar{a} \delta_{A} D \beta', positive=True)
tau = kx**2*t
defs = [sy.Eq(a, eO + eA*(1 + sy.cos(theta)**2)/2),               # mean damping
        sy.Eq(d, eA*sy.sin(theta)**2/2),                            # ambipolar anisotropy
        sy.Eq(beta, eH*sy.cos(theta)),                              # Hall rotation
        sy.Eq(D**2, d**2 - beta**2)]                                # root of the characteristic equation
rates = {eO: a - d*(1 + sy.cos(theta)**2)/sy.sin(theta)**2,
         eA: 2*d/sy.sin(theta)**2,
         eH: sy.sqrt(d**2 - D**2)/sy.cos(theta)}
def sub_rates(e):                                                   # substitute, write sin^2 as 1 - cos^2, simplify
    return sy.simplify(sy.factor(e.subs(rates).subs(sy.sin(theta)**2, 1 - sy.cos(theta)**2)))
B_rates = [sy.Eq(s.lhs, sy.simplify(sy.trigsimp(s.rhs.subs(rates)))) for s in B_sol]
display(*B_rates)


# 3) beta back in, and the exponentials as cosh and sinh
def to_hyp(f):                                             # e^{a tau} f = P cosh(D tau) + Q sinh(D tau)/D
    g = sy.simplify(f*sy.exp(a*tau))
    P = sy.simplify(g.subs(t, 0))                          # value at t = 0
    Q = sy.simplify(sy.diff(g, t).subs(t, 0)/kx**2)        # tau-slope at t = 0
    h = sy.exp(-a*tau)*(P*sy.cosh(D*tau) + Q*sy.sinh(D*tau)/D)
    assert sy.simplify((h - f).rewrite(sy.exp)) == 0       # identical to its input
    return h
B_hyp = [sy.Eq(s.lhs, to_hyp(s.rhs.subs(sy.sqrt(d**2 - D**2), beta))) for s in B_rates]
display(*B_hyp)
B_sol_full = sy.simplify(sy.Matrix([0, list(B_hyp)[0].rhs*sy.cos(kx*x), list(B_hyp)[1].rhs*sy.cos(kx*x)]) + B0*b_hat) # add background field back
display(Math(r'B(r,t) = ' + sy.latex(B_sol_full)))

display(Math(r'\nabla \cdot B(r,0) = ' + sy.latex(div(B))))
display(Math(r'\nabla \cdot B(r,t) = ' + sy.latex(div(B_sol_full))))
#%%
# =============================================================================== case 3
banner("3. PARABOLIC COUPLED SYSTEM (Thermal + Magnetic + Nernst/Ettinghausen")
theta = sy.symbols(r'\theta', real=True)
eO, eA, B0, = sy.symbols(r'\eta_{O} \eta_{A} B_{0}', positive=True)
dBy = sy.Function(r'\delta B_{y}')(t)
dBz = sy.Function(r'\delta B_{z}')(t)
T0 = sy.symbols(r'T_{0}', positive=True)

kn, kpe, kjw = sy.symbols(r'\kappa_{n} \kappa_{\perp} \kappa_{J\wedge}', positive=True)
dT = sy.Function(r'\delta T')(t)
T = T0 + dT*sy.cos(kx*x)
gT = grad(T)
b_hat = sy.Matrix([0, 1, 0]) # B0 = |B0|y_hat
B = B0*b_hat + sy.Matrix([0, dBy, 0])*sy.cos(kx*x)  # background + transverse perturbation
J = curl(B)

E = eO*J + eA*(J - J.dot(b_hat)*b_hat) - (1/T0)*xi*kjw * (b_hat.cross(gT)) # Ohmic + ambipolar + Nernst
q = (gam-1)*rho0*((1-xi)*kn*gT + xi*kpe*(gT - gT.dot(b_hat)*b_hat) + (xi*kjw/rho0)*(b_hat.cross(J)))
divq = div(q)


LHS = sy.diff(sy.Matrix([T,B]),t)
RHS = sy.Matrix([div(q)/rho0, -curl(E)])

ode = sy.simplify(LHS - RHS)
display(Math(sy.latex(ode) + r' = 0'))

cx = sy.cos(kx*x)
rhs_T = sy.expand(sy.solve(ode[0]/cx, sy.diff(dT, t))[0])     # dT/dt = ...
rhs_B = sy.expand(sy.solve(ode[2]/cx, sy.diff(dBy, t))[0])     # dB/dt = ...
DT, DB, cTB, cBT = sy.symbols(r'D_{T} D_{B} c_{TB} c_{BT}', positive=True)
coef = {DT:  -rhs_T.coeff(dT)/kx**2,  cTB: -rhs_T.coeff(dBy)/kx**2,
        cBT: -rhs_B.coeff(dT)/kx**2,  DB:  -rhs_B.coeff(dBy)/kx**2}
display(*[sy.Eq(k_, sy.factor(v_)) for k_, v_ in coef.items()])

odes = [sy.diff(dT, t) + kx**2*(DT*dT + cTB*dBy), sy.diff(dBy, t) + kx**2*(cBT*dT + DB*dBy)]
assert sy.simplify(odes[0].subs(coef) - (sy.diff(dT, t) - rhs_T)) == 0
assert sy.simplify(odes[1].subs(coef) - (sy.diff(dBy, t) - rhs_B)) == 0
dT0, dB0, dBz0 = sy.symbols(r'\delta{T}_{0} \delta{B}_{0} \delta{B}_{z0}', real=True)
X_sol = sy.dsolve(odes, [dT, dBy], ics={dT.subs(t, 0): dT0, dBy.subs(t, 0): dB0})
display(*X_sol)

# 2) substitutions afterwards: mean rate, half-difference, and D
a, D = sy.symbols(r'\bar{a} D', positive=True)
d = sy.symbols(r'd', real=True)
tau = kx**2*t
defs = [sy.Eq(a, (DT + DB)/2), sy.Eq(d, (DT - DB)/2), sy.Eq(D**2, d**2 + cTB*cBT)]
rates = {DT: a + d, DB: a - d, cBT: (D**2 - d**2)/cTB}
display(Math(r"\text{definitions:}"), *defs)
display(Math(r"\text{substitutions:}"), *[sy.Eq(k_, v_) for k_, v_ in rates.items()])
display(Math(r"\text{check:}\ " + sy.latex([sy.simplify((e_.lhs - e_.rhs).subs(rates)) for e_ in defs])))
roots = sorted({p.base for s in X_sol for p in s.rhs.atoms(sy.Pow) if p.exp in (sy.S.Half, -sy.S.Half)},
               key=sy.default_sort_key)
for r_ in roots:
    display(Math(sy.latex(sy.sqrt(r_)) + r"\ \to\ " + sy.latex(sy.simplify(sy.sqrt(r_).subs(rates)))))
args = sorted({e_.args[0] for s in X_sol for e_ in s.rhs.atoms(sy.exp)}, key=sy.default_sort_key)
for g_ in args:
    display(Math(sy.latex(g_) + r"\ \to\ " + sy.latex(sy.factor(sy.simplify(g_.subs(rates))))))
X_rates = [sy.Eq(s.lhs, sy.simplify(s.rhs.subs(rates))) for s in X_sol]

# 3) exponentials as cosh and sinh, printing the coefficients read off
def to_hyp(f, name):                                       # e^{a tau} f = P cosh(D tau) + Q sinh(D tau)/D
    g = sy.simplify(f*sy.exp(a*tau))
    P = sy.simplify(g.subs(t, 0))                          # value at t = 0
    Q = sy.simplify(sy.diff(g, t).subs(t, 0)/kx**2)        # tau-slope at t = 0
    h = sy.exp(-a*tau)*(P*sy.cosh(D*tau) + Q*sy.sinh(D*tau)/D)
    assert sy.simplify((h - f).rewrite(sy.exp)) == 0
    P, Q = [sy.simplify(c_.subs(D, sy.sqrt(d**2 + cTB*cBT))) for c_ in (P, Q)]   # D^2 - delta^2 back to cTB cBT
    return sy.exp(-a*tau)*(P*sy.cosh(D*tau) + Q*sy.sinh(D*tau)/D)
X_hyp = [sy.Eq(s.lhs, to_hyp(s.rhs, sy.latex(s.lhs))) for s in X_rates]
display(*X_hyp)

# 4) full fields and div B
T_sol = T.subs(dT, X_hyp[0].rhs)
B_sol = B.subs({dBy: X_hyp[1].rhs})
display(Math(r"T(\mathbf{r},t) = " + sy.latex(T_sol)), Math(r"\mathbf{B}(\mathbf{r},t) = " + sy.latex(B_sol)))
display(Math(r"\nabla\cdot\mathbf{B} = " + sy.latex(sy.simplify(div(B_sol)))))

#%%
# =============================================================================== case 4
banner("4. Righi-Leduc")
kw = sy.symbols(r'\kappa_{\wedge}')
k_T, k_B, U = sy.symbols(r'k_{T} k_{B} U', positive=True)
kRL, T0, B0, dT0 = sy.symbols(r'\kappa_{\wedge} T_{0} B_{0} \delta{T}_{0}', positive=True)
C, S = sy.Function('C')(t), sy.Function('S')(t) 

b_hat = sy.Matrix([sy.cos(k_B*z), sy.sin(k_B*z), 0]) # Beltrami: curl[B] = k|B| b, div B = 0
B = B0*b_hat                                               # |B| = B0 uniform, static (v = 0, no field terms)

T = T0 + C*sy.cos(k_T*x) + S*sy.sin(k_T*x) # General form

gT = grad(T)
q = (gam-1)*xi*kRL*b_hat.cross(gT)                   # Righi-Leduc flux
ode = sy.expand(sy.simplify(sy.diff(T, t) - div(q)))

cos_part = ode.coeff(sy.cos(k_T*x))
sin_part = ode.coeff(sy.sin(k_T*x))
assert sy.simplify(ode - cos_part*sy.cos(k_T*x) - sin_part*sy.sin(k_T*x)) == 0

odes = [cos_part, sin_part]
trig_sol = sy.dsolve([odes[0], odes[1]], [C, S])

C1, C2 = sy.symbols('C1 C2')
ics = {C.subs(t, 0): dT0, S.subs(t, 0): 0} # T(x, 0) = T0 + dT0 cos(k_T x)
consts = sy.solve([s.rhs.subs(t, 0) - ics[s.lhs.subs(t, 0)] for s in trig_sol], [C1, C2], dict=True)[0]
display(consts)
trig_sol = [sy.Eq(s.lhs, s.rhs.subs(consts)) for s in trig_sol]
display(*trig_sol)
T_sol_full = sy.simplify(T.subs({C: trig_sol[0].rhs, S: trig_sol[1].rhs}))
T_sol_full = T_sol_full.replace(sy.cos, lambda a: sy.cos(sy.collect(sy.expand(a), t, sy.factor))) # tidy
display(Math(r"T(\mathbf{r},t) = " + sy.latex(T_sol_full)))

#%%
# =============================================================================== case 5
banner("5. GYROVISCOSITY")
nw1, nw2, v0, kz = sy.symbols(r'\nu_{\wedge1} \nu_{\wedge2} v_{0} k_{z}', positive=True)
LC = sy.LeviCivita
bxM = lambda b, M: sy.Matrix(3, 3, lambda i, j: sum(LC(i, k, l)*b[k]*M[l, j] for k in range(3) for l in range(3)))  # (b x M)_ij
Mxb = lambda M, b: sy.Matrix(3, 3, lambda i, j: sum(LC(j, k, l)*M[i, k]*b[l] for k in range(3) for l in range(3)))  # (M x b)_ij
b_hat = sy.Matrix([0, 0, 1])                                     # B0 along z

# 0) Velocity pertubation
dv_init = sy.Matrix([v0*sy.sin(kz*z), 0, v0*sy.sin(kx*x)]) # one mode across b, one along b
display(Math(r"\mathbf{v}(0) = " + sy.latex(dv_init)))

# general velocity
basis = [sy.sin(kx*x), sy.cos(kx*x), sy.sin(kz*z), sy.cos(kz*z)]
F = sy.Matrix(3, 4, lambda j, m: sy.Function('F_{%s%d}' % ('xyz'[j], m))(t))
v = sy.Matrix([sum(F[j, m]*basis[m] for m in range(4)) for j in range(3)])
display(Math(r"\mathbf{v}(t) = " + sy.latex(v)))


gv = sy.Matrix(3, 3, lambda i, j: sy.diff(v[j], (x, y, z)[i]))   # (grad v)_ij = d_i v_j
divv = div(v)
W = gv + gv.T - sy.Rational(2, 3)*divv*sy.eye(3)
bb = b_hat*b_hat.T
Ip = sy.eye(3) - bb
Pi3 = -(xi*rho0*nw1/2)*(bxM(b_hat, W*Ip) - Mxb(Ip*W, b_hat)) # flux += +0.5 HII rho nu_wedge1 (b x W.Ip - Ip.W x b)
Pi4 = -2*xi*rho0*nw2*(bxM(b_hat, W*bb) - Mxb(bb*W, b_hat))   # flux += +2 HII rho nu_wedge2 (b x W.bb - bb.W x b)
divPi = div_tensor(Pi3+Pi4)
ode = sy.expand(rho0*sy.diff(v, t) - divPi) # rho0 dv/dt = div(Pi)

for j in range(3):
    assert sy.simplify(ode[j] - sum(ode[j].coeff(f_)*f_ for f_ in basis)) == 0
odes = [sy.collect(sy.expand(ode[j].coeff(basis[m])/rho0), list(F), sy.factor) for m in range(4) for j in range(3)]
ics = {F[j, m].subs(t, 0): sy.expand(dv_init[j]).coeff(basis[m]) for j in range(3) for m in range(4)}
sol = sy.dsolve(odes, list(F), ics=ics)
display(*[s for s in sol if s.rhs != 0])  # the amplitudes that come out non-zero
display(Math(r"\text{zero for all } t:\ " + sy.latex([s.lhs for s in sol if s.rhs == 0])))

#substitutions afterwards: the two rotation rates
W3, W4 = sy.symbols(r'\Omega_{3} \Omega_{4}', positive=True)
defs = [sy.Eq(W3, xi*nw1*kx**2), sy.Eq(W4, 2*xi*nw2*kz**2)]
rates = {nw1: W3/(xi*kx**2), nw2: W4/(2*xi*kz**2)}
display(Math(r"\text{definitions:}"), *defs)
sol = [sy.Eq(s.lhs, sy.simplify(s.rhs.subs(rates))) for s in sol]

# full velocity solution
v_sol = sy.simplify(v.subs({s.lhs: s.rhs for s in sol}))
display(Math(r"\mathbf{v}(\mathbf{r},t) = " + sy.latex(v_sol)))

#%%
# =============================================================================== case 6
banner("6. HYPERBOLIC COUPLED SYSTEM (Hall + Peltier/Seebeck")
theta = sy.symbols(r'\theta', positive=True)
T0, B0, dT0 = sy.symbols(r'T_{0} B_{0} \delta{T}_{0}', positive=True)
kpa, kpe, eH = sy.symbols(r'\kappa_{J\parallel} \kappa_{J\perp} \eta_{H}', positive=True)
b_hat = sy.Matrix([sy.cos(theta), sy.sin(theta), 0])

# 0) initial perturbation (temperature only)
dT_init = dT0*sy.cos(kx*x)
dB_init = sy.Matrix([0, 0, 0])
B = B0*b_hat
display(Math(r"T(0) = " + sy.latex(T0 + dT_init)))
display(Math(r"\mathbf{B}(0) = " + sy.latex(B)))

# general T and B
basis = [sy.cos(kx*x), sy.sin(kx*x)]
FT = [sy.Function('T_{%d}' % m)(t) for m in range(2)] ## general form: cos*dT0 + sin*dT1
FB = sy.Matrix(3, 2, lambda j, m: sy.Function('B_{%s%d}' % ('xyz'[j], m))(t)) ## general form: cos*dBx0, cos*dBy0 + ...
T = T0 + sum(FT[m]*basis[m] for m in range(2))
B +=  sy.Matrix([sum(FB[j, m]*basis[m] for m in range(2)) for j in range(3)])
J, gT = curl(B), grad(T)

# flux and field
q = xi*kpa*(b_hat.dot(J))*b_hat + xi*kpe*(J - b_hat.dot(J)*b_hat) # Peltier
E = eH*J.cross(b_hat) - xi*(1/T0)*kpa*(b_hat.dot(gT))*b_hat - xi*(1/T0)*kpe*(gT - (b_hat.dot(gT))*b_hat)  # Hall + Seebeck
q = q*(gam-1)/rho0

LHS = sy.diff(sy.Matrix([T,B]),t)
RHS = sy.Matrix([div(q), -curl(E)])

ode = sy.simplify(LHS - RHS)
display(Math(sy.latex(ode) + r' = 0'))

ode = sy.expand(sy.simplify(LHS - RHS))
odes = [ode[i].coeff(f_) for f_ in basis for i in range(4)]     # each row split into its cos and sin parts: 8 equations
ics = {FB[n, m].subs(t, 0): 0 for n in range(3) for m in range(2)} # dB is 0 at start
ics.update({FT[0].subs(t, 0): dT0, FT[1].subs(t, 0): 0}) # dT_0(0) = dT0, dT_1(0) = 0 
sol = sy.dsolve(odes, list(FT) + list(FB), ics=ics)
display(Math(sy.latex(sol)))

# cos rows form a 3x3 system for (T_0, B_y0, B_z0); its eigenvalues are 0 and +-i Omega kx^2
lam, W = sy.symbols(r'\lambda \Omega', real=True)
A = sy.Matrix(3, 3, lambda i, j: -ode[(0, 2, 3)[i]].coeff(basis[0]).coeff((FT[0], FB[1, 0], FB[2, 0])[j])/kx**2)
W2 = sy.simplify(sy.cancel(-(A - lam*sy.eye(3)).det()/lam) - lam**2) # det = -lambda (lambda^2 + Omega^2)
display(Math(r"\Omega^2 = " + sy.latex(sy.collect(sy.expand(W2), [eH], sy.factor))))

#each amplitude is f(0) + c1 (cos(Omega kx^2 t) - 1) + c2 sin(Omega kx^2 t)/Omega: read c1, c2 off at t = 0
def to_trig(f):
    c2 = sy.factor(sy.cancel(sy.diff(f, t).subs(t, 0)/kx**2))                 # slope, per kx^2
    c1 = sy.factor(sy.cancel(-sy.diff(f, t, 2).subs(t, 0)/(kx**4*W**2)))      # curvature, per kx^4
    return sy.simplify(f.subs(t, 0)) + c1*(sy.cos(W*kx**2*t) - 1) + c2*sy.sin(W*kx**2*t)/W
sol_t = [sy.Eq(s.lhs, to_trig(s.rhs)) for s in sol]
display(*[s for s in sol_t if s.rhs != 0])

# full solution
sub = {s.lhs: s.rhs for s in sol_t}
T_sol, B_sol = T.subs(sub), B.subs(sub)
display(Math(r"T(\mathbf{r},t) = " + sy.latex(T_sol)), Math(r"\mathbf{B}(\mathbf{r},t) = " + sy.latex(B_sol)))
