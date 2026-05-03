'''Generate HTCondor job list for main sweep.

Three conditions:
    biofilm  : chainModel with l=100
    chainWM  : chainModel with l=2000 (no core forms)
    wellMixed: explicit well-mixed model

chainWM and wellMixed should give statistically equivalent results. Running
both is a sanity check -- any meaningful disagreement is a signal of a bug
in either model.

Writes sweepMainJobs.txt: one line per job with (condition, dose, seedStart).
'''
import numpy as np

excess = np.logspace(np.log10(0.05), np.log10(2.0), 20)
doses = np.round(1.0 + excess, 4)

conditions = ['biofilm', 'chainWM', 'wellMixed']
seedsPerBatch = 200
batchesPerPoint = 5  # 1000 seeds per (condition, dose)

with open('sweepMainJobs.txt', 'w') as f:
    for cond in conditions:
        for dose in doses:
            for batch in range(batchesPerPoint):
                seedStart = batch * seedsPerBatch
                f.write(f'{cond}, {dose}, {seedStart}\n')

nJobs = len(conditions) * len(doses) * batchesPerPoint
print(f'Wrote {nJobs} jobs to sweepMainJobs.txt')
print(f'Conditions: {conditions}')
print(f'Doses: {len(doses)} log-spaced from {doses[0]} to {doses[-1]}')