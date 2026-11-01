# test of local vs normal wavespeed for whistler waves

import numpy as np
rng = np.random.default_rng(2)
worst_tot = worst_nrm = 0.0
for _ in range(300_000):
    ca2, cs2 = 10**rng.uniform(-3, 3, 2)
    E = 10**rng.uniform(-4, 4); u = rng.uniform(0, 1)
    cf2 = ca2 + cs2
    fast2 = 0.5*(cf2 + np.sqrt(cf2**2 - 4*ca2*cs2*u))           # local fast speed^2
    r = np.roots([1.0, -(cf2 + u*(ca2 + E)),
                  u*(ca2*(ca2 + 2*cs2) + E*cs2), -ca2**2*cs2*u**2]).real.max()
    cw_tot = 0.5*(np.sqrt(E) + np.sqrt(E + 4*ca2))               # total Alfven
    cw_nrm = 0.5*(np.sqrt(E) + np.sqrt(E + 4*ca2*u))             # normal Alfven
    worst_tot = max(worst_tot, r/max(fast2, cw_tot**2))
    worst_nrm = max(worst_nrm, r/max(fast2, cw_nrm**2))
print(f"Hall only, local c_f + total  c_a in c_w: {worst_tot:.6f}")
print(f"Hall only, local c_f + normal c_a in c_w: {worst_nrm:.6f}")