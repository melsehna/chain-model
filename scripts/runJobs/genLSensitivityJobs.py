'''Generate sensitivityLJobs.txt for sweepSensitivityL.sub.

biofilm condition only; sweeps edge width l x 8 doses x 5 batches.
narrative.tex Section 8 lists l (edge width) and core size N_0 - l as
crossover-determining parameters; this sweep tests that prediction.

l=100 is the main-sweep default; l=25 and l=400 are extremes that should
shift the crossover dose if the analytical framework is correct.
'''

import itertools

doses          = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
l_vals         = [25, 50, 100, 200, 400]
seeds_per_job  = 200
total_seeds    = 5000

with open('sensitivityLJobs.txt', 'w') as f:
    for dose, lVal in itertools.product(doses, l_vals):
        for seed_start in range(0, total_seeds, seeds_per_job):
            f.write(f'{dose} {seed_start} {lVal}\n')

n_jobs = len(doses) * len(l_vals) * (total_seeds // seeds_per_job)
print(f'Wrote {n_jobs} jobs to sensitivityLJobs.txt')
print(f'  {len(doses)} doses x {len(l_vals)} l values x {total_seeds // seeds_per_job} batches')
