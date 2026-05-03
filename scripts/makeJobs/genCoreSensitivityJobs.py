'''Generate HTCondor job list for core dormancy sensitivity.

Tests how biofilm rescue depends on how dormant the core is. Only runs
l=100 (biofilm); the well-mixed comparison uses the main sweep's l=2000 data.
'''
import numpy as np

excess = np.logspace(np.log10(0.05), np.log10(2.0), 15)
doses = np.round(1.0 + excess, 4)

bCoreRatios = [0.05, 0.1, 0.2]  # core 20x, 10x, 5x slower than edge
seedsPerBatch = 200
batchesPerPoint = 3

with open('sensitivityCoreJobs.txt', 'w') as f:
    for bCore in bCoreRatios:
        for dose in doses:
            for batch in range(batchesPerPoint):
                seedStart = batch * seedsPerBatch
                f.write(f'{dose}, {seedStart}, {bCore}\n')

nJobs = len(bCoreRatios) * len(doses) * batchesPerPoint
print(f'Wrote {nJobs} jobs to sensitivityCoreJobs.txt')