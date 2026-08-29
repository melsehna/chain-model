'''Generate args for the density-regime comparison at l = 50.

Two doses (low, high) x three arms x two density regimes x 500 seeds,
in batches of 100. Answers whether the biofilm's low-dose deficit is a
property of the Wilson density gradient or survives without it.

Arms:
  bf      biofilm, l = 50, b_c = 0.2   (active core)
  bfdorm  biofilm, l = 50, b_c = 0     (dormant core; supply equals well-mixed
                                        exactly, so any deficit is establishment)
  wm      chainWM, l = 2000            (no core; well-mixed limit of the chain)

Regimes:
  K = 1000 (= nInit)  current behavior: R births scaled by max(0, 1 - N/K),
                      so core R is subcritical and no R grows until N < 400
  K = 1e9             factor is ~1 throughout: core R critical, edge R at
                      s_R = 0.4 from t = 0

Regimes:
  N0    K = 1000, linear factor (current behavior)
  off   K = 1e9,  linear factor with K far above N (no cap in practice)
  step  K = 1000, ceiling only: no effect below K, births blocked at it

Writes densityCompareJobs.txt:
  condition dose seedStart bCore lVal kVal densityForm kTag tag
'''

doses         = [1.05, 2.8]
seeds_total   = 500
seeds_per_job = 100
arms = [
    ('bf',     'biofilm', 0.2, 50),
    ('bfdorm', 'biofilm', 0.0, 50),
    ('wm',     'chainWM', 0.2, 2000),
]
# (kTag, K, densityForm)
regimes = [('N0', 1000.0, 'linear'), ('off', 1e9, 'linear'), ('step', 1000.0, 'step')]

n = 0
with open('densityCompareJobs.txt', 'w') as f:
    for tag, condition, bcore, lval in arms:
        for dose in doses:
            for kTag, kVal, dForm in regimes:
                for s in range(0, seeds_total, seeds_per_job):
                    f.write(f'{condition} {dose} {s} {bcore} {lval} {kVal:g} {dForm} {kTag} {tag}\n')
                    n += 1
print(f'densityCompareJobs.txt: {n} jobs '
      f'({len(arms)} arms x {len(doses)} doses x {len(regimes)} regimes x '
      f'{seeds_total // seeds_per_job} batches of {seeds_per_job} seeds)')
