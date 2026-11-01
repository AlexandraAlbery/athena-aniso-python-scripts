# Numerical test of local vs normal cf for Peltier/Seebeck (+Hall) thermoelectric wave

import numpy as np
rng = np.random.default_rng(1)
worst_B = worst_E = 0.0
NORMAL_ALFVEN = True   # set False if AlfvenSpeed() returns |B|/sqrt(rho)
for _ in range(300_000):
    ca2, cs2 = 10**rng.uniform(-3, 3, 2)
    gam = rng.uniform(1.01, 3.0); ciso2 = cs2/gam
    D, E = 10**rng.uniform(-4, 4, 2)
    s = rng.choice([-1.0, 1.0]); u = rng.uniform(0, 1)
    cf2 = ca2 + cs2
    fast2 = 0.5*(cf2 + np.sqrt(cf2**2 - 4*ca2*cs2*u))     # code: normal-direction fast speed^2
    # field: Hall + Peltier/Seebeck cubic at this angle
    A2 = -(cf2 + u*(ca2 + E) + D*u*(1 - u))
    A1 = (ca2*u*(ca2 + 2*cs2) + E*cs2*u
          + u*(1 - u)*(D*(ca2 + ciso2) + 2*s*np.sqrt(ca2*ciso2*(gam - 1)*D*E)))
    A0 = -ca2*ciso2*u**2*(gam*ca2 + D*(1 - u))
    rB = np.roots([1.0, A2, A1, A0]).real.max()
    cax2 = ca2*u if NORMAL_ALFVEN else ca2
    cw = 0.5*(np.sqrt(E) + np.sqrt(E + 4*cax2))
    worst_B = max(worst_B, rB/(max(fast2, cw**2) + D/4 + np.sqrt(D*E)/2))
    # internal energy: Peltier/Seebeck cubic vs code's c_TE at this angle
    B2 = -(cf2 + ca2*u + D*u*(1 - u))
    B1 = ca2*u*(ca2 + 2*cs2) + D*u*(1 - u)*(ca2 + ciso2)
    B0 = -ca2*ciso2*u**2*(gam*ca2 + D*(1 - u))
    rE = np.roots([1.0, B2, B1, B0]).real.max()
    worst_E = max(worst_E, rE/(fast2 + D/4))
print(f"field : max root / code estimate = {worst_B:.6f}")
print(f"energy: max root / code estimate = {worst_E:.6f}")