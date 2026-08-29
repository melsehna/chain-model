'''Generate args for the theta-targeted tracer grid.

Delivery is a per-lineage outcome that should depend only on

    theta = b_c * (N0 - l) / (s_e * l)

so the test is to hit each theta target by several different routes through
(b_c, l, dose) and check they agree, and to compare against the two candidate
closed forms:

    simple    ln(1 + th) / th
    refined   2 * ((1 + th) * ln(1 + th) - th) / th^2

Runs use tracer mode: R takes the WT rates everywhere and rescue is disabled,
so nothing selects on the marker, the population trajectory is that of a pure
WT run, and no run is truncated. Both are required, since delivery measured in
rescue-truncated runs is censored and delivery measured under the linear
density factor is suppressed (core R is subcritical there).

Core lineages per run is mu * (N0 - l) * theta / 2, so mu is set per point to
land near a target lineage count rather than being held fixed. The marker is
neutral, so mu has no effect on the delivery probability itself; it only sets
how many labels exist. Labelling never depletes the WT pool appreciably here:
the labelling timescale ln2/(mu*b_c) exceeds the Phase 1 duration T1 by orders
of magnitude at every point below.

Columns: condition dose seedStart mu bCore lVal thetaTag routeTag
'''

import math

N0            = 1000
thetas        = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
routes        = [(0.05, 100), (0.2, 100), (0.8, 100),
                 (0.2, 25), (0.2, 50), (0.2, 200), (0.2, 400),
                 (0.05, 400), (0.4, 200), (0.05, 200)]
seeds_total   = 1000
seeds_per_job = 200
DOSE_MIN, DOSE_MAX = 1.02, 6.0
TARGET_LINEAGES    = 2000
MU_MIN, MU_MAX     = 1e-5, 0.2
MAX_ROUTES         = 3

rows, summary = [], []
for th in thetas:
    picked = 0
    for bc, lVal in routes:
        if picked >= MAX_ROUTES:
            break
        s_e = bc * (N0 - lVal) / (th * lVal)
        dose = 1.0 + s_e
        if not (DOSE_MIN <= dose <= DOSE_MAX):
            continue
        # core lineages per run = mu * (N0 - l) * theta / 2
        mu = TARGET_LINEAGES / (seeds_total * (N0 - lVal) * th / 2)
        mu = min(max(mu, MU_MIN), MU_MAX)
        tag = f'bc{bc}l{lVal}'
        for s in range(0, seeds_total, seeds_per_job):
            rows.append(('biofilm', round(dose, 4), s, mu, bc, lVal, f'th{th:g}', tag))
        expected = mu * (N0 - lVal) * th / 2 * seeds_total
        summary.append((th, bc, lVal, round(dose, 3), mu, int(expected)))
        picked += 1

with open('thetaGridJobs.txt', 'w') as f:
    for condition, dose, s, mu, bc, lVal, thTag, tag in rows:
        f.write(f'{condition} {dose} {s} {mu:g} {bc} {lVal} {thTag} {tag}\n')

print(f'{"theta":>7}{"b_c":>6}{"l":>5}{"dose":>7}{"mu":>10}{"lineages":>10}'
      f'{"simple":>9}{"refined":>9}')
for th, bc, lVal, dose, mu, exp in summary:
    print(f'{th:>7g}{bc:>6}{lVal:>5}{dose:>7.3f}{mu:>10.2g}{exp:>10}'
          f'{math.log(1+th)/th:>9.3f}{2*((1+th)*math.log(1+th)-th)/th**2:>9.3f}')
print(f'\nthetaGridJobs.txt: {len(rows)} jobs, '
      f'{len(summary)} parameter points, {seeds_total} seeds each')
