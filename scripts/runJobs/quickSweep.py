'''Quick local sweep across all 3 conditions.

Verifies the sweep pipeline and checks chainWM/wellMixed agreement before
committing to OSG. Takes ~6 min on a workstation.
'''
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

doses = np.array([1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8])
seedsPerPoint = 200
conditions = ['biofilm', 'chainWM', 'wellMixed']
mu = 1e-4

CONDITION_COLORS = {
    'biofilm':   'C0',
    'chainWM':   'C1',
    'wellMixed': 'C2',
}

allRows = []
t0 = time.perf_counter()
for cond in conditions:
    for dose in doses:
        subprocess.run([
            'python', 'runBatch.py',
            '--condition', cond, '--dose', str(dose),
            '--seedStart', '0', '--seedCount', str(seedsPerPoint),
            '--mu', str(mu), '--bCoreRatio', '0.2',
            '--maxTime', '200',
            '--out', f'quick_{cond}_dose{dose:.2f}.csv',
        ], check=True, capture_output=True)
        df = pd.read_csv(f'quick_{cond}_dose{dose:.2f}.csv')
        allRows.append(df)
        elapsed = time.perf_counter() - t0
        print(f'{cond:>10s} dose={dose:.2f} rescued={df.rescued.sum()}/{len(df)} '
              f'[{elapsed:.0f}s]')

df = pd.concat(allRows, ignore_index=True)
df.to_csv('quick_sweep.csv', index=False)

summary = (df.groupby(['condition', 'dose'])
             .agg(nRuns=('seed', 'count'),
                  nRescued=('rescued', 'sum'))
             .reset_index())
summary['pRescue'] = summary.nRescued / summary.nRuns
summary['pRescueSE'] = np.sqrt(summary.pRescue * (1 - summary.pRescue) / summary.nRuns)
print()
print(summary.to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
for cond in conditions:
    sub = summary[summary.condition == cond].sort_values('dose')
    ax.errorbar(sub.dose, sub.pRescue, yerr=sub.pRescueSE,
                marker='o', label=cond, capsize=3,
                color=CONDITION_COLORS[cond])
ax.set_xlabel('d_WtEdge (dose)')
ax.set_ylabel('P(rescue)')
ax.set_title(f'Rescue probability, mu={mu}, {seedsPerPoint} seeds/point')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
biofilmRescued = df[(df.condition == 'biofilm') & (df.rescued == 1)]
pathway = (biofilmRescued.groupby('dose').rescueMode
           .value_counts(normalize=True)
           .unstack(fill_value=0).sort_index())
if 'core' in pathway.columns:
    ax.plot(pathway.index, pathway['core'], 'o-',
            label='core delivery', color='C3')
if 'edge' in pathway.columns:
    ax.plot(pathway.index, pathway['edge'], 'o-',
            label='edge mutation', color='C4')
ax.set_xlabel('d_WtEdge')
ax.set_ylabel('Fraction of biofilm rescues')
ax.set_title('Pathway breakdown (biofilm)')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)

# Agreement check
ax = axes[2]
cwm = summary[summary.condition == 'chainWM'].sort_values('dose')
wm = summary[summary.condition == 'wellMixed'].sort_values('dose')
merged = cwm[['dose', 'pRescue', 'pRescueSE']].merge(
    wm[['dose', 'pRescue', 'pRescueSE']],
    on='dose', suffixes=('_chainWM', '_wellMixed'))
merged['diff'] = merged.pRescue_chainWM - merged.pRescue_wellMixed
merged['diffSE'] = np.sqrt(merged.pRescueSE_chainWM**2
                           + merged.pRescueSE_wellMixed**2)
ax.errorbar(merged.dose, merged['diff'], yerr=merged.diffSE,
            marker='o', capsize=3, color='black')
ax.axhline(0, color='red', ls='--', alpha=0.5)
ax.set_xlabel('d_WtEdge')
ax.set_ylabel('P(rescue)_chainWM - P(rescue)_wellMixed')
ax.set_title('Agreement check\n(should scatter around 0)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('quick_sweep.png', dpi=120)
print()
print(f'Total time: {time.perf_counter() - t0:.0f}s')