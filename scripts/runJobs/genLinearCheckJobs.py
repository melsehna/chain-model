'''Generate args for the post-mortem on the old density rule.

The claim to be tested: under densityForm='linear' with K = nInit, R births are
suppressed by the *global* population size, so the biofilm's protected core mass
docks the mutant's growth rate and rescue is gated behind N < K(1 - dR/bR) = 400.
The size of that penalty should scale with how long the population sits above the
gate, which is set by the Phase 1 decline rate s_e * l:

    time above the gate = (N0 - 400) / (s_e * l)

so the deficit should be largest at small l, shrink as l grows, and vanish for
chainWM (no core ever forms; decline is exponential, as in well-mixed). At high
dose the gate is cleared almost immediately at every l, so no deficit anywhere.

The dormant core (b_c = 0) is used throughout: it contributes no core-born
mutations, so supply matches well-mixed exactly and any difference in rescue is
per-mutation conversion, with nothing to confound it.

Everything here runs at densityForm='linear' (the old rule) with kEst = 8; the
matched step-rule numbers are already on disk as fxCore / fxMain. The well-mixed
reference is re-run rather than taken from main.csv so that seeds, kEst and the
column set match.

Columns: condition dose seedStart bCore lVal tag
'''

import itertools

seeds_total   = 5000
seeds_per_job = 200
batches       = range(0, seeds_total, seeds_per_job)

rows = []
# Dose 1.05: where the artifact was largest. Full l ladder plus both controls.
for lVal, s in itertools.product([25, 50, 100, 200, 400], batches):
    rows.append(('biofilm', 1.05, s, 0.0, lVal, f'bc0l{lVal}'))
for s in batches:
    rows.append(('chainWM', 1.05, s, 0.0, 2000, 'chainWM'))
    rows.append(('wellMixed', 1.05, s, 0.0, 100, 'wellMixed'))

# Dose 2.8: control. Gate is cleared at once, so no deficit at any l.
for lVal, s in itertools.product([25, 100, 400], batches):
    rows.append(('biofilm', 2.8, s, 0.0, lVal, f'bc0l{lVal}'))
for s in batches:
    rows.append(('wellMixed', 2.8, s, 0.0, 100, 'wellMixed'))

with open('linearCheckJobs.txt', 'w') as f:
    for condition, dose, s, bc, lVal, tag in rows:
        f.write(f'{condition} {dose} {s} {bc} {lVal} {tag}\n')

from collections import Counter
for k, v in sorted(Counter((r[1], r[5]) for r in rows).items()):
    print(f'  dose {k[0]:<5} {k[1]:<10} {v} jobs')
print(f'linearCheckJobs.txt: {len(rows)} jobs, {seeds_total} seeds per point')
