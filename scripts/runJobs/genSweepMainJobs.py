'''Generate sweepMainJobs.txt for sweepMain.sub.

3 conditions x 8 doses x 25 batches = 600 jobs, 5000 seeds per point.
'''

import itertools

doses = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
conditions = ['biofilm', 'chainWM', 'wellMixed']
seeds_per_job = 200
total_seeds   = 5000

with open('sweepMainJobs.txt', 'w') as f:
    for condition, dose in itertools.product(conditions, doses):
        for seed_start in range(0, total_seeds, seeds_per_job):
            f.write(f'{condition} {dose} {seed_start}\n')

n_jobs = len(conditions) * len(doses) * (total_seeds // seeds_per_job)
print(f'Wrote {n_jobs} jobs to sweepMainJobs.txt')
print(f'  {len(conditions)} conditions × {len(doses)} doses × {total_seeds // seeds_per_job} batches')
