'''Generate HTCondor job list for mutation rate sensitivity.

Runs all three conditions (biofilm, chainWM, wellMixed) for each mu value,
so we get the biofilm-vs-well-mixed comparison across mutation regimes AND
the chainWM/wellMixed agreement check at each mu.
'''
import numpy as np

excess = np.logspace(np.log10(0.05), np.log10(2.0), 15)
doses = np.round(1.0 + excess, 4)

conditions = ['biofilm', 'chainWM', 'wellMixed']
mus = [1e-6, 1e-5, 1e-4, 1e-3]
seedsPerBatch = 200
batchesPerPoint = 3

with open('sensitivityMuJobs.txt', 'w') as f:
    for cond in conditions:
        for mu in mus:
            for dose in doses:
                for batch in range(batchesPerPoint):
                    seedStart = batch * seedsPerBatch
                    f.write(f'{cond}, {dose}, {seedStart}, {mu}\n')

nJobs = len(conditions) * len(mus) * len(doses) * batchesPerPoint
print(f'Wrote {nJobs} jobs to sensitivityMuJobs.txt')