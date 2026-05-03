'''Test statistical equivalence: chain model with l >> N_init vs well-mixed.

If the chain model's core never forms (because l is always larger than N),
then it should behave identically to a well-mixed population with the same
per-cell rates. This test verifies that numerically.
'''
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
from chainModel import simulateChain
from wellMixed import simulateWellMixed


def paramsChainNoCore(dose, mu=1e-4, lVal=2000):
    '''Chain model parameters with l large enough that core never forms.'''
    return {
        'mu': mu, 'nInit': 1000, 'l': lVal,
        'bWtCore': 0.2, 'dWtCore': 0.2,  # irrelevant since no core forms
        'bWtEdge': 1.0, 'dWtEdge': dose,
        'bREdge': 1.0, 'dREdge': 0.6,
    }


def paramsWellMixed(dose, mu=1e-4):
    '''Well-mixed parameters matching the chain edge rates.'''
    return {
        'mu': mu, 'nInit': 1000,
        'bWtEdge': 1.0, 'dWtEdge': dose,
        'bREdge': 1.0, 'dREdge': 0.6,
    }


def collectStats(simulator, params, nSeeds=200, maxTime=100.0):
    nRescued = 0
    nExtinct = 0
    rescueTimes = []
    extinctionTimes = []
    finalNs = []
    modes = {'core': 0, 'edge': 0}
    for seed in range(nSeeds):
        r = simulator(params, seed=seed, maxTime=maxTime, stopAtRescue=True)
        if r['rescued']:
            nRescued += 1
            rescueTimes.append(r['rescueTime'])
            if r['rescueMode']:
                modes[r['rescueMode']] = modes.get(r['rescueMode'], 0) + 1
        if r['extinct']:
            nExtinct += 1
            extinctionTimes.append(r['extinctionTime'])
        finalNs.append(r['finalN'])
    return {
        'pRescue': nRescued / nSeeds,
        'pExtinct': nExtinct / nSeeds,
        'meanRescueT': np.mean(rescueTimes) if rescueTimes else None,
        'meanExtinctT': np.mean(extinctionTimes) if extinctionTimes else None,
        'modes': modes,
    }


print(f"{'dose':>6} {'metric':<20} {'chain (l=2000)':>18} {'wellMixed':>15}")
print("-" * 66)

for dose in [1.1, 1.3, 1.6, 2.0]:
    cParams = paramsChainNoCore(dose)
    wParams = paramsWellMixed(dose)
    cStats = collectStats(simulateChain, cParams, nSeeds=300)
    wStats = collectStats(simulateWellMixed, wParams, nSeeds=300)

    for key, label in [('pRescue', 'P(rescue)'),
                       ('pExtinct', 'P(extinct)'),
                       ('meanRescueT', 'mean rescue time'),
                       ('meanExtinctT', 'mean extinct time')]:
        cv = cStats[key]
        wv = wStats[key]
        cvs = f'{cv:.3f}' if isinstance(cv, float) else str(cv)
        wvs = f'{wv:.3f}' if isinstance(wv, float) else str(wv)
        print(f"{dose:>6.2f} {label:<20} {cvs:>18} {wvs:>15}")
    print()