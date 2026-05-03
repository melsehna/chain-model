'''Extra kymograph: find a soft-sweep rescue (multiple lineages contributing).

Searches seeds for a case where nLineagesAtRescueEdge >= 3 so we get
visually distinct colors for multiple lineages in the rescue.
'''

import sys, os

import matplotlib.pyplot as plt
from chainModel import simulateChain
from kymograph import plotKymograph


params = {
    'mu': 5e-3, 'nInit': 1000, 'l': 100,
    'bWtCore': 0.1, 'dWtCore': 0.1,
    'bWtEdge': 0.5, 'dWtEdge': 1.0,
    'bREdge':  0.5, 'dREdge':  0.3,
}

best = None
for seed in range(100):
    r = simulateChain(params, seed=seed, maxTime=60.0, recordEvery=10)
    if r['rescued'] and r['nLineagesAtRescueEdge'] >= 3:
        best = (seed, r)
        print(f'seed={seed}: nLinAtRescue={r["nLineagesAtRescueEdge"]} '
              f'primary={r["primaryLineage"]} primCount={r["primaryLineageCount"]} '
              f'origin={r["primaryLineageOrigin"]} '
              f'appearedCore={r["nLineagesAppearedCore"]} appearedEdge={r["nLineagesAppearedEdge"]}')
        break

if best is None:
    print('no multi-lineage rescue found')
    sys.exit(0)

seed, r = best
fig, ax = plt.subplots(figsize=(10, 7))
plotKymograph(r, params, f'soft-sweep rescue (seed={seed}, {r["nLineagesAtRescueEdge"]} lineages)', ax=ax)
plt.tight_layout()
os.makedirs('figures/', exist_ok=True)
plt.savefig('figures/kymo-softSweep.png', dpi=110, bbox_inches='tight')
print('Saved figures/kymo-softSweep.png')
plt.close()