'''Generate sensitivityCoreJobs.txt for sweepSensitivityCore.sub.

biofilm condition only; sweeps bCoreRatio × 8 doses × 5 batches.
bCore=0.0 is the dormant-core baseline; higher values test active-core effects.
'''

import itertools

doses       = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
bcore_vals  = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
seeds_per_job = 200
total_seeds   = 5000

with open('sensitivityCoreJobs.txt', 'w') as f:
    for dose, bcore in itertools.product(doses, bcore_vals):
        for seed_start in range(0, total_seeds, seeds_per_job):
            f.write(f'{dose} {seed_start} {bcore}\n')

n_jobs = len(doses) * len(bcore_vals) * (total_seeds // seeds_per_job)
print(f'Wrote {n_jobs} jobs to sensitivityCoreJobs.txt')
print(f'  {len(doses)} doses × {len(bcore_vals)} bCore values × {total_seeds // seeds_per_job} batches')
