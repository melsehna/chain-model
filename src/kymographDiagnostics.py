'''End-to-end diagnostic of the chainModel + kymograph pipeline.

Produces four kymographs illustrating the main dynamical regimes:
  1. Pre-treatment steady state (no drug effect, biofilm at equilibrium)
  2. Super-lethal extinction (drug kills edge faster than R can arise)
  3. Edge-mutation rescue (R mutation born in the edge establishes locally)
  4. Core-delivery rescue (R mutation born in core drifts outward and establishes)

The pre-treatment and extinction plots use maxTime + no rescue filter.
The two rescue plots scan seeds until they find a clean example of each mode.
'''

import sys
import os
import matplotlib.pyplot as plt


from chainModel import simulateChain
from kymograph import plotKymograph


def params(dWtEdge, mu, dREdge=0.3):
    return {
        'mu': mu, 'nInit': 1000, 'l': 100,
        'bWtCore': 0.1, 'dWtCore': 0.1,
        'bWtEdge': 0.5, 'dWtEdge': dWtEdge,
        'bREdge':  0.5, 'dREdge':  dREdge,
    }


# 1. Pre-treatment: drug off (dE = bE). Steady state.
p1 = params(dWtEdge=0.5, mu=1e-4, dREdge=0.5)
r1 = simulateChain(p1, seed=42, maxTime=40.0, recordEvery=80)

# 2. Super-lethal: drug strong, mutation rate low. Biofilm extincts without rescue.
p2 = params(dWtEdge=1.0, mu=1e-6, dREdge=0.3)
r2 = simulateChain(p2, seed=42, maxTime=40.0, recordEvery=30)

# 3. Edge-mutation rescue. Higher mu so mutations arise faster in the edge.
# Scan seeds for a clean edge-origin rescue where the biofilm survives meaningfully.
p3 = params(dWtEdge=1.0, mu=2e-4, dREdge=0.3)
r3 = None
for seed in range(200):
    r = simulateChain(p3, seed=seed, maxTime=60.0, recordEvery=10)
    if (r['rescued']
            and r['primaryLineageOrigin'] == 'edge'
            and r['finalN'] > 200):
        r3 = r
        print(f'edge-mutation rescue: seed={seed} rescueTime={r["rescueTime"]:.2f} '
              f'nLinAtRescue={r["nLineagesAtRescueEdge"]}')
        break

# 4. Core-delivery rescue. Harder to find -- stronger drug to push biofilm back
# toward the core, giving core-born mutations time to drift out.
p4 = params(dWtEdge=1.2, mu=5e-4, dREdge=0.3)
r4 = None
for seed in range(500):
    r = simulateChain(p4, seed=seed, maxTime=60.0, recordEvery=10)
    if (r['rescued']
            and r['primaryLineageOrigin'] == 'core'
            and 3 < r['rescueTime'] < 30
            and r['finalN'] > 100):
        r4 = r
        print(f'core-delivery rescue: seed={seed} rescueTime={r["rescueTime"]:.2f} '
              f'nLinAtRescue={r["nLineagesAtRescueEdge"]} '
              f'primary born at t={r["primaryLineageBirthTime"]:.2f}')
        break

if r3 is None:
    print('WARNING: no edge-mutation rescue found in 200 seeds')
if r4 is None:
    print('WARNING: no core-delivery rescue found in 500 seeds')

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plotKymograph(r1, p1, 'pre-treatment (dE = bE = 0.5)', ax=axes[0, 0])
plotKymograph(r2, p2, 'super-lethal extinction (dE = 1.0, mu = 1e-6)', ax=axes[0, 1])
if r3:
    plotKymograph(r3, p3, 'edge-mutation rescue (dE = 1.0, mu = 2e-4)', ax=axes[1, 0])
else:
    axes[1, 0].text(0.5, 0.5, 'no edge-mutation rescue found', ha='center', va='center',
                    transform=axes[1, 0].transAxes)
if r4:
    plotKymograph(r4, p4, 'core-delivery rescue (dE = 1.2, mu = 5e-4)', ax=axes[1, 1])
else:
    axes[1, 1].text(0.5, 0.5, 'no core-delivery rescue found', ha='center', va='center',
                    transform=axes[1, 1].transAxes)

plt.tight_layout()
os.makedirs('figures/', exist_ok=True)
plt.savefig('figures/kymographDiagnostics.png', dpi=300, bbox_inches='tight')
print('Saved figures/kymographDiagnostics.png')
plt.close()