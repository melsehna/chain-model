'''Generate sensitivityMuJobs.txt for sweepSensitivityMu.sub.

3 conditions x 3 mu values x 8 doses x 25 batches = 1800 jobs, 5000 seeds per point.
'''

import itertools

doses         = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
conditions    = ['biofilm', 'chainWM', 'wellMixed']
mu_vals       = [1e-5, 1e-4, 1e-3]
seeds_per_job = 200
total_seeds   = 5000

with open('sensitivityMuJobs.txt', 'w') as f:
    for condition, dose, mu in itertools.product(conditions, doses, mu_vals):
        for seed_start in range(0, total_seeds, seeds_per_job):
            f.write(f'{condition} {dose} {seed_start} {mu}\n')

n_jobs = len(conditions) * len(doses) * len(mu_vals) * (total_seeds // seeds_per_job)
print(f'Wrote {n_jobs} jobs to sensitivityMuJobs.txt')
print(f'  {len(conditions)} conditions × {len(doses)} doses × {len(mu_vals)} mu values × {total_seeds // seeds_per_job} batches')
