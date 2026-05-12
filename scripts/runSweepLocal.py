'''Multiprocess driver for biofilm rescue parameter sweeps.

Runs the chain (biofilm) model and the well-mixed control in-process across
parameter grids. Writes one CSV per sweep with all per-sim observables.

Sweeps (phase portraits):
  phaseBc : (dose x b_c)  -- 6 b_c x 12 doses x 2 conditions x N seeds
  phaseL  : (dose x l)    -- 5 l values x 12 doses x 1 condition (chain) + WM x N seeds
  phaseMu : (dose x mu)   -- 4 mu x 12 doses x 2 conditions x N seeds

Dose grid: 12 log-spaced points in s_e = d_e - 1 from 0.005 to 2.0
(denser near MIC, sparser at strong selection).

Density regulation: both wellMixed and chainModel use Wilson's logistic
factor on R births with K = nInit. WT births stay density-independent.
This is Wilson 2017's "high-density rescue" regime; post-rescue equilibrium
of R is K * (1 - d_R/b_R).

Seeds: each sim uses np.random.default_rng(seed). Seeds run 0..N-1 and are
reused across parameter points (paired-by-seed across conditions).

Usage:
    python3 scripts/runSweepLocal.py --sweep phaseBc --seedsPerPoint 2500
    python3 scripts/runSweepLocal.py --sweep phaseL  --seedsPerPoint 2500
    python3 scripts/runSweepLocal.py --sweep phaseMu --seedsPerPoint 2500

The maxTime per sim is set adaptively per dose: t_max = max(200, 50/s_e),
which gives Phase 1 a long enough window to complete at low s_e.
'''

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from chainModel import simulateChain
from wellMixed import simulateWellMixed


# Dose grid: log-spaced in s_e, denser near MIC.
SE_GRID = np.logspace(np.log10(0.005), np.log10(2.0), 12)
DOSE_GRID = np.round(1.0 + SE_GRID, 6)

# Default condition geometry
L_DEFAULT = 100
N_INIT = 1000
B_C_DEFAULT = 0.2
MU_DEFAULT = 1e-4

# Phase grids
BC_GRID = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
L_GRID = [10, 30, 100, 300]
'''
Note: l = N_INIT (fully-mixed limit of the chain model) is omitted from
the phaseL grid because it duplicates the wellMixed control included in
the same sweep, but is much slower (no Phase 2 transition; entire
population is edge for the full sim). Use the wellMixed condition as
the l -> N_INIT reference instead.
'''
MU_GRID = [1e-6, 1e-5, 1e-4, 1e-3]


CSV_HEADER = [
    'seed', 'condition', 'dose', 'mu', 'bCoreRatio', 'l',
    'rescued', 'rescueTime', 'rescueMode',
    'extinct', 'extinctionTime',
    'wtExtinct', 'wtExtinctTime',
    'finalN', 'terminationReason',
    'nMutCore', 'nMutEdge',
    'nLineagesEstablished',
    'nLineagesEstablishedCore', 'nLineagesEstablishedEdge',
    'nLineagesAtRescue',
]


def buildChainParams(dose, mu, bCoreRatio, lVal):
    return {
        'mu': mu,
        'nInit': N_INIT,
        'l': lVal,
        'bWtCore': bCoreRatio,
        'dWtCore': bCoreRatio,
        'bWtEdge': 1.0,
        'dWtEdge': dose,
        'bREdge': 1.0,
        'dREdge': 0.6,
        'K': float(N_INIT),
    }


def buildWellMixedParams(dose, mu):
    return {
        'mu': mu,
        'nInit': N_INIT,
        'bWtEdge': 1.0,
        'dWtEdge': dose,
        'bREdge': 1.0,
        'dREdge': 0.6,
        'K': float(N_INIT),
    }


def adaptiveLimits(dose):
    '''Per-dose maxTime and maxGenerations.

    maxTime: enough for Phase 1 + Phase 2 to resolve. Phase 1 takes
        (N0-l)/(s_e l) time, Phase 2 about log(l)/s_e -- both scale as 1/s_e.

    maxGenerations: events-per-time scales with population size (~N0). Need
        N0 * maxTime events as a safety factor.
    '''
    s_e = max(dose - 1.0, 1e-6)
    maxTime = float(max(200.0, 50.0 / s_e))
    maxGenerations = max(500_000, int(2 * N_INIT * maxTime))
    return maxTime, maxGenerations


def runOne(task):
    condition, dose, mu, bCoreRatio, lVal, seed = task
    maxTime, maxGenerations = adaptiveLimits(dose)

    if condition == 'biofilm':
        params = buildChainParams(dose, mu, bCoreRatio, lVal)
        r = simulateChain(params, seed=seed, maxTime=maxTime,
                          maxGenerations=maxGenerations, stopAtRescue=True)
        nMutCore = r['nLineagesAppearedCore']
        nMutEdge = r['nLineagesAppearedEdge']
        nLinEstCore = r['nLineagesEstablishedCore']
        nLinEstEdge = r['nLineagesEstablishedEdge']
        nLinEstTotal = r['nLineagesEstablished']
        nLinAtRescue = r.get('nLineagesAtRescueEdge')
        bcOut = bCoreRatio
    elif condition == 'wellMixed':
        params = buildWellMixedParams(dose, mu)
        r = simulateWellMixed(params, seed=seed, maxTime=maxTime,
                              maxGenerations=maxGenerations, stopAtRescue=True)
        nMutCore = 0
        nMutEdge = r['nLineagesAppeared']
        nLinEstCore = 0
        nLinEstEdge = r['nLineagesEstablished']
        nLinEstTotal = r['nLineagesEstablished']
        nLinAtRescue = r.get('nLineagesAtRescue')
        bcOut = ''
    else:
        raise ValueError(f'unknown condition: {condition}')

    return (
        seed, condition, dose, mu, bcOut, lVal,
        int(r['rescued']),
        r['rescueTime'] if r['rescueTime'] is not None else '',
        r['rescueMode'] if r['rescueMode'] is not None else '',
        int(r['extinct']),
        r['extinctionTime'] if r['extinctionTime'] is not None else '',
        int(r['wtExtinct']),
        r['wtExtinctTime'] if r['wtExtinctTime'] is not None else '',
        r['finalN'],
        r['terminationReason'],
        nMutCore, nMutEdge,
        nLinEstTotal, nLinEstCore, nLinEstEdge,
        nLinAtRescue if nLinAtRescue is not None else '',
    )


def buildTaskList(sweep, seedsPerPoint):
    tasks = []
    if sweep == 'phaseBc':
        # (dose x b_c) for biofilm; WM has no b_c so one curve only.
        for dose in DOSE_GRID:
            for bc in BC_GRID:
                for seed in range(seedsPerPoint):
                    tasks.append(('biofilm', float(dose), MU_DEFAULT, bc, L_DEFAULT, seed))
            for seed in range(seedsPerPoint):
                tasks.append(('wellMixed', float(dose), MU_DEFAULT, B_C_DEFAULT, L_DEFAULT, seed))
    elif sweep == 'phaseL':
        # (dose x l) for biofilm; WM is the l = nInit limit (separate condition).
        for dose in DOSE_GRID:
            for lVal in L_GRID:
                for seed in range(seedsPerPoint):
                    tasks.append(('biofilm', float(dose), MU_DEFAULT, B_C_DEFAULT, lVal, seed))
            for seed in range(seedsPerPoint):
                tasks.append(('wellMixed', float(dose), MU_DEFAULT, B_C_DEFAULT, L_DEFAULT, seed))
    elif sweep == 'phaseMu':
        for dose in DOSE_GRID:
            for mu in MU_GRID:
                for seed in range(seedsPerPoint):
                    tasks.append(('biofilm', float(dose), mu, B_C_DEFAULT, L_DEFAULT, seed))
                for seed in range(seedsPerPoint):
                    tasks.append(('wellMixed', float(dose), mu, B_C_DEFAULT, L_DEFAULT, seed))
    elif sweep == 'calibration':
        # Small sweep at the weak-selection corner to confirm rescue is estimable.
        cornerDoses = DOSE_GRID[:4]
        for dose in cornerDoses:
            for bc in [0.0, 0.2]:
                for seed in range(seedsPerPoint):
                    tasks.append(('biofilm', float(dose), MU_DEFAULT, bc, L_DEFAULT, seed))
            for seed in range(seedsPerPoint):
                tasks.append(('wellMixed', float(dose), MU_DEFAULT, B_C_DEFAULT, L_DEFAULT, seed))
    else:
        raise ValueError(f'unknown sweep: {sweep}')
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', required=True,
                   choices=['phaseBc', 'phaseL', 'phaseMu', 'calibration'])
    p.add_argument('--seedsPerPoint', type=int, default=2500)
    p.add_argument('--nWorkers', type=int, default=48)
    p.add_argument('--chunksize', type=int, default=8)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    resultsDir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(resultsDir, exist_ok=True)
    outPath = args.out or os.path.join(resultsDir, f'{args.sweep}.csv')

    tasks = buildTaskList(args.sweep, args.seedsPerPoint)
    nTasks = len(tasks)
    print(f'sweep={args.sweep}  seedsPerPoint={args.seedsPerPoint}  '
          f'nWorkers={args.nWorkers}  nTasks={nTasks:,}')
    print(f'doses (s_e): {[round(s, 4) for s in SE_GRID]}')
    print(f'out={outPath}')

    t0 = time.perf_counter()
    nDone = 0
    progressEvery = max(1, nTasks // 50)
    nextProgress = progressEvery

    with open(outPath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        with Pool(args.nWorkers) as pool:
            for row in pool.imap_unordered(runOne, tasks, chunksize=args.chunksize):
                w.writerow(row)
                nDone += 1
                if nDone >= nextProgress:
                    elapsed = time.perf_counter() - t0
                    rate = nDone / elapsed
                    eta = (nTasks - nDone) / rate if rate > 0 else 0
                    print(f'  {nDone:>9,}/{nTasks:,}  '
                          f'elapsed={elapsed:6.0f}s  '
                          f'rate={rate:7.0f}/s  '
                          f'eta={eta:6.0f}s', flush=True)
                    nextProgress += progressEvery

    elapsed = time.perf_counter() - t0
    print(f'Done. {nTasks:,} sims in {elapsed:.0f}s '
          f'({nTasks/elapsed:.0f}/s). Wrote {outPath}')


if __name__ == '__main__':
    main()
