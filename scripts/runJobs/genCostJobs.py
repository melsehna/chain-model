'''Generate args for the core-only cost-of-resistance sweep.

R is favoured where the drug is and penalised where it is not: the cost is
expressed only in the drug-free core,

    bRCore = bWtCore * (1 - c),   dRCore = dWtCore,

so a core R cell decays at rate s_core = b_c * c while the edge rates (the
resistant phenotype under drug) are untouched. Well-mixed pays nothing because
it has no drug-free compartment; the asymmetry between conditions is produced
by geometry alone.

Two channels this is meant to separate:

  delivery   a core-born lineage must survive s_core until the boundary reaches
             it, a wait of order T1 = (N0 - l)/(s_e l). The penalty exponent is
             c * theta with theta = b_c * T1, which runs 36 (dose 1.05) to 1.0
             (dose 2.8) at b_c = 0.2, so the same cost is ~36x more punishing at
             low dose. This should erase the biofilm's low-dose supply bonus.

  churn      an edge-born lineage is repeatedly pushed back into the core by the
             conveyor (every edge birth reclassifies one edge cell as core). With
             a neutral core that is free; with a cost it is not. This is the only
             channel that can push the biofilm below well-mixed, and it is
             measured directly by nEdgeBornEnteredCore / nSweepsEdgeBorn.

Predictions: the deficit appears only for active cores and scales with b_c; a
dormant core (b_c = 0) is structurally immune, since bRCore = dRCore = 0 means
no core dynamics at all. chainWM is a null control: no core ever forms, so the
cost must do nothing.

c = 0 is not generated: those points already exist as the fxCore sweep, which
ran at the same K, densityForm and kEst. The well-mixed reference is fxMain.

Columns: condition dose seedStart mu bCore lVal costR tag
'''

import itertools
import sys

PILOT = '--pilot' in sys.argv
PROBE = '--probe' in sys.argv

doses         = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
seeds_total   = 5000
seeds_per_job = 200
batches       = range(0, seeds_total, seeds_per_job)
MU            = 1e-4

# (label, condition, bCore, l, costs to run)
arms = [
    ('bc0.2',   'biofilm', 0.2, 100,  [0.01, 0.03, 0.1, 0.3]),  # default active core
    ('bc0.8',   'biofilm', 0.8, 100,  [0.01, 0.03, 0.1, 0.3]),  # penalty should scale with b_c
    ('bc0',     'biofilm', 0.0, 100,  [0.03, 0.3]),             # structurally immune
    ('chainWM', 'chainWM', 0.2, 2000, [0.1]),                   # null control, no core
]

# Pilot: the two doses that bracket the predicted effect (c*theta = 1.08 vs 0.03
# at c = 0.03), enough arms to check that the penalty scales with b_c and that a
# dormant core is untouched. Confirms the effect and the cost ladder's centring
# before the full grid is committed.
# Probe: the pilot showed the penalty is far weaker than exp(-c*theta) predicts
# (at c = 0.1, low dose, delivery fell 0.132 -> 0.114 rather than to 0.004),
# because the lineages that get delivered are the shallow, recently born ones
# that barely wait. Only the strongest combination came close to parity
# (b_c = 0.8, c = 0.1 -> 0.98 at dose 1.05). This probe pushes the cost well past
# anything biologically plausible, at the core activity most likely to cross, to
# find whether the ratio crosses 1 at all and at which dose it comes back up.
if PROBE:
    arms = [
        ('bc0.8', 'biofilm', 0.8, 100, [0.3, 0.5], [1.05, 1.2, 1.6, 2.8]),
    ]
elif PILOT:
    arms = [
        ('bc0.2',   'biofilm', 0.2, 100, [0.03, 0.1], [1.05, 2.8]),
        ('bc0.8',   'biofilm', 0.8, 100, [0.1],       [1.05, 2.8]),
        ('bc0',     'biofilm', 0.0, 100, [0.03],      [1.05]),
    ]
else:
    arms = [(label, cond, bc, lVal, costs, doses)
            for label, cond, bc, lVal, costs in arms]

rows = []
for label, condition, bc, lVal, costs, armDoses in arms:
    for c, dose, s in itertools.product(costs, armDoses, batches):
        rows.append((condition, dose, s, MU, bc, lVal, c, f'{label}c{c:g}'))

outName = ('costProbeJobs.txt' if PROBE else
           'costPilotJobs.txt' if PILOT else 'costJobs.txt')
with open(outName, 'w') as f:
    for condition, dose, s, mu, bc, lVal, c, tag in rows:
        f.write(f'{condition} {dose} {s} {mu:g} {bc} {lVal} {c:g} {tag}\n')

from collections import Counter
counts = Counter(r[7].rsplit('c', 1)[0] for r in rows)
for k in sorted(counts):
    print(f'  {k}: {counts[k]} jobs')
print(f'{outName}: {len(rows)} jobs, {seeds_total} seeds per (arm, cost, dose) point')
