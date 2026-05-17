'''Generate args files for the Figure 4 Panel B targeted re-sweep.

Boosts the three populations that feed Figure 4 Panel B (and Figure 3) from
5,000 to 20,000 seeds per (condition, dose) combination, by adding seeds
5000-19799 in batches of 200. Writes two args files:

  panel4BMain.txt       — wellMixed × 8 doses × 75 batches = 600 jobs.
                          Output files land under main_ prefix in results/.
  panel4BSensCore.txt   — biofilm at bCore ∈ {0.0, 0.2} × 8 doses × 75 batches
                          = 1200 jobs. Output files land under sensCore_ prefix.

Total: 1800 jobs. Filenames already include seedStart, so the new outputs
land alongside the existing 0-4799 seed batches in scripts/runJobs/results/
and are picked up automatically by aggregateResults.py on re-aggregation.
'''

import itertools

doses          = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
seeds_per_job  = 200
seed_start     = 20000    # previous boost covered 5000-19999
seed_end       = 100000   # target 100k total per combo
n_batches      = (seed_end - seed_start) // seeds_per_job

# main.csv data: just wellMixed. The biofilm-at-bCore=0.2 line in main.csv
# isn't used by the dose curve / decomposition plots (those pull the active
# biofilm from sensCore.csv instead), so we don't need to boost it.
with open('panel4BMain.txt', 'w') as f:
    for dose in doses:
        for s in range(seed_start, seed_end, seeds_per_job):
            f.write(f'wellMixed {dose} {s}\n')

n_main = len(doses) * n_batches
print(f'panel4BMain.txt: {n_main} jobs (wellMixed only)')

# sensCore.csv data: biofilm at bCore ∈ {0.0, 0.2}.
with open('panel4BSensCore.txt', 'w') as f:
    for dose, bcore in itertools.product(doses, [0.0, 0.2]):
        for s in range(seed_start, seed_end, seeds_per_job):
            f.write(f'{dose} {s} {bcore}\n')

n_sc = len(doses) * 2 * n_batches
print(f'panel4BSensCore.txt: {n_sc} jobs (biofilm at bCore=0.0 and 0.2)')

print(f'Total: {n_main + n_sc} jobs')
