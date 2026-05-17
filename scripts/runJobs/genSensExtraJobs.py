'''Generate args files for the §6 sensitivity-sweep targeted boost.

Boosts the three sensitivity sweeps (sensL, sensCore, sensMu) from 5,000
to 20,000 seeds per (parameter, dose) combination by adding seeds
5000-19799 in batches of 200. Writes three args files:

  sensLExtraJobs.txt      — biofilm × 8 doses × 5 l-values × 75 batches
                            = 3000 jobs.
  sensCoreExtraJobs.txt   — biofilm × 8 doses × 4 b_c-values × 75 batches
                            = 2400 jobs.
                            Excludes b_c ∈ {0.0, 0.2} which were already
                            boosted to 100k seeds for Figure 4 Panel B.
  sensMuExtraJobs.txt     — 3 conditions × 8 doses × 3 mu-values × 75
                            batches = 5400 jobs.

Total: 10,800 jobs. Output files include seedStart in the filename, so
the new files land alongside the existing 0-4799 seed batches in
scripts/runJobs/results/ and are picked up automatically by
aggregateResults.py on re-aggregation.
'''

import itertools

doses          = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
seeds_per_job  = 200
seed_start     = 5000     # existing runs cover 0-4999
seed_end       = 20000    # target 20k total per combo
n_batches      = (seed_end - seed_start) // seeds_per_job

# sensL: biofilm × dose × l values.
l_vals = [25, 50, 100, 200, 400]
with open('sensLExtraJobs.txt', 'w') as f:
    for dose, lVal in itertools.product(doses, l_vals):
        for s in range(seed_start, seed_end, seeds_per_job):
            f.write(f'{dose} {s} {lVal}\n')
n_l = len(doses) * len(l_vals) * n_batches
print(f'sensLExtraJobs.txt: {n_l} jobs')

# sensCore: biofilm × dose × b_c values (excluding 0.0 and 0.2 already boosted).
bcore_vals = [0.05, 0.1, 0.4, 0.8]
with open('sensCoreExtraJobs.txt', 'w') as f:
    for dose, bcore in itertools.product(doses, bcore_vals):
        for s in range(seed_start, seed_end, seeds_per_job):
            f.write(f'{dose} {s} {bcore}\n')
n_sc = len(doses) * len(bcore_vals) * n_batches
print(f'sensCoreExtraJobs.txt: {n_sc} jobs')

# sensMu: condition × dose × mu values.
conditions = ['biofilm', 'chainWM', 'wellMixed']
mu_vals    = [1e-5, 1e-4, 1e-3]
with open('sensMuExtraJobs.txt', 'w') as f:
    for condition, dose, mu in itertools.product(conditions, doses, mu_vals):
        for s in range(seed_start, seed_end, seeds_per_job):
            f.write(f'{condition} {dose} {s} {mu}\n')
n_mu = len(conditions) * len(doses) * len(mu_vals) * n_batches
print(f'sensMuExtraJobs.txt: {n_mu} jobs')

print(f'Total: {n_l + n_sc + n_mu} jobs')
