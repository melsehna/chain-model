'''Generate args for the full re-run under the corrected model.

Corrected model = densityForm 'step' (K acts as a ceiling on R births rather
than as a gradient across the whole density range) with K = nInit, plus
kEst = 8 for the count-based establishment criterion. The exact criterion
(lineage never went extinct) is recorded alongside it as nSurvived*.

One args file covers all four sweeps; the prefix/tag columns keep the output
filenames separated so aggregateResults.py can glob them independently:

  fxMain   3 conditions x 8 doses                      (b_c = 0.2, l = 100)
  fxCore   6 core activities x 8 doses                 (l = 100)
  fxL      5 edge widths x 8 doses                     (b_c = 0.2)
  fxMu     3 conditions x 3 mutation rates x 8 doses   (b_c = 0.2, l = 100)

Columns: condition dose seedStart mu bCore lVal prefix tag
'''

import itertools

doses         = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
seeds_total   = 5000
seeds_per_job = 200
batches       = range(0, seeds_total, seeds_per_job)

L_BIOFILM, L_CHAINWM = 100, 2000
rows = []

def lFor(condition, lVal=L_BIOFILM):
    return L_CHAINWM if condition == 'chainWM' else lVal

# fxMain: the three conditions at default parameters.
for condition, dose, s in itertools.product(['biofilm', 'chainWM', 'wellMixed'], doses, batches):
    rows.append((condition, dose, s, 1e-4, 0.2, lFor(condition), 'fxMain', condition))

# fxCore: core activity sweep, biofilm only.
for bc, dose, s in itertools.product([0.0, 0.05, 0.1, 0.2, 0.4, 0.8], doses, batches):
    rows.append(('biofilm', dose, s, 1e-4, bc, L_BIOFILM, 'fxCore', f'bc{bc}'))

# fxL: edge width sweep, biofilm only.
for lVal, dose, s in itertools.product([25, 50, 100, 200, 400], doses, batches):
    rows.append(('biofilm', dose, s, 1e-4, 0.2, lVal, 'fxL', f'l{lVal}'))

# fxMu: mutation rate sweep, all three conditions.
for condition, mu, dose, s in itertools.product(['biofilm', 'chainWM', 'wellMixed'],
                                                [1e-5, 1e-4, 1e-3], doses, batches):
    rows.append((condition, dose, s, mu, 0.2, lFor(condition), 'fxMu', f'{condition}mu{mu:g}'))

with open('fixedSweepJobs.txt', 'w') as f:
    for condition, dose, s, mu, bc, lVal, prefix, tag in rows:
        f.write(f'{condition} {dose} {s} {mu:g} {bc} {lVal} {prefix} {tag}\n')

from collections import Counter
counts = Counter(r[6] for r in rows)
for k in ('fxMain', 'fxCore', 'fxL', 'fxMu'):
    print(f'  {k}: {counts[k]} jobs')
print(f'fixedSweepJobs.txt: {len(rows)} jobs total, {seeds_total} seeds per parameter point')
