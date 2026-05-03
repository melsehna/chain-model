'''Explicit well-mixed control model.

A well-mixed bacterial population with the same birth/death/mutation rules
as the chain model's edge compartment, but no spatial structure. Used as an
independent cross-check on the chain model's well-mixed limit (l >= nInit).

The chain model with l >= nInit should give statistically equivalent results
to this model -- see testEquivalence.py. Running both in the sweep provides
a consistency check: any meaningful disagreement is a signal of a bug.
'''

import numpy as np


def sampleEvent(rng, ratesList):
    total = 0.0
    for _, r in ratesList:
        total += r
    if total <= 0.0:
        return None, np.inf
    dt = rng.exponential(1.0 / total)
    threshold = rng.random() * total
    acc = 0.0
    for name, r in ratesList:
        acc += r
        if threshold <= acc:
            return name, dt
    return ratesList[-1][0], dt


def pickLineage(rLineages, rTotal, rng):
    '''Return a lineage id sampled proportional to its live count in rLineages.
    rTotal must equal sum(rLineages.values()).'''
    if rTotal <= 0:
        raise RuntimeError('pickLineage called with rTotal=0')
    threshold = rng.random() * rTotal
    acc = 0
    lastLin = None
    for lin, cnt in rLineages.items():
        acc += cnt
        lastLin = lin
        if threshold < acc:
            return lin
    # Fall through due to float rounding: return last key
    return lastLin


def simulateWellMixed(params, seed=None, maxGenerations=200_000,
                      maxTime=np.inf, nMax=None, stopAtRescue=False,
                      rThreshold=10):
    '''Run the well-mixed control.

    Parameters (dict):
        mu         mutation rate per birth
        nInit      initial population size
        bWtEdge    WT birth rate
        dWtEdge    WT death rate (sweep variable)
        bREdge     R birth rate
        dREdge     R death rate

    Rescue: r >= rThreshold (default 10).

    Lineage tracking: every R lineage that ever appears is recorded in the
    internal `lineages` dict. Aggregate counts over this dict are returned.
    '''
    rng = np.random.default_rng(seed)

    mu    = float(params['mu'])
    nInit = int(params['nInit'])
    bWt   = float(params['bWtEdge'])
    dWt   = float(params['dWtEdge'])
    bR    = float(params['bREdge'])
    dR    = float(params['dREdge'])

    if nMax is None:
        nMax = 10 * nInit

    wt = nInit
    # Live R lineage counts; entries are dropped when count hits 0 (metadata
    # is kept in `lineages`). r = sum(rLineages.values()).
    rLineages = {}
    r = 0
    N = wt + r

    time = 0.0
    generations = 0

    rescued = False
    rescueTime = None
    rescueGeneration = None
    rescueRLineages = None  # snapshot of rLineages at rescue trigger

    nextLineage = 2
    mutationEvents = []

    # Per-lineage metadata. Entries are never removed.
    lineages = {}

    terminationReason = None

    while True:
        if generations >= maxGenerations:
            terminationReason = 'maxGenerations'
            break
        if time >= maxTime:
            terminationReason = 'maxTime'
            break
        if N <= 0:
            terminationReason = 'extinction'
            break
        if N > nMax:
            terminationReason = 'nMaxExceeded'
            break

        ratesList = (
            ('birthWt', bWt * wt),
            ('birthR',  bR  * r),
            ('deathWt', dWt * wt),
            ('deathR',  dR  * r),
        )

        event, dt = sampleEvent(rng, ratesList)
        if event is None:
            terminationReason = 'zeroRate'
            break

        time += dt
        generations += 1

        if event == 'birthWt':
            if rng.random() < mu:
                lin = nextLineage
                rLineages[lin] = 1
                r += 1
                lineages[lin] = {
                    'liveCount': 1,
                    'birthTime': time,
                    'birthGeneration': generations,
                    'deathTime': None,
                    'deathGeneration': None,
                }
                mutationEvents.append({
                    'lineage': lin,
                    'time': time,
                    'generation': generations,
                })
                nextLineage += 1
            else:
                wt += 1
            N += 1
        elif event == 'birthR':
            lin = pickLineage(rLineages, r, rng)
            rLineages[lin] += 1
            r += 1
            lineages[lin]['liveCount'] += 1
            N += 1
        elif event == 'deathWt':
            wt -= 1
            N -= 1
        elif event == 'deathR':
            lin = pickLineage(rLineages, r, rng)
            rLineages[lin] -= 1
            r -= 1
            info = lineages[lin]
            info['liveCount'] -= 1
            if rLineages[lin] == 0:
                del rLineages[lin]
                info['deathTime'] = time
                info['deathGeneration'] = generations
            N -= 1

        if (not rescued) and r >= rThreshold:
            rescued = True
            rescueTime = time
            rescueGeneration = generations
            # Snapshot for correctness when stopAtRescue=False
            rescueRLineages = dict(rLineages)
            if stopAtRescue:
                terminationReason = 'rescue'
                break

    if terminationReason is None:
        terminationReason = 'unknown'

    extinct = (N == 0)

    # ----- Aggregate lineage statistics -----
    nLineagesAppeared = len(lineages)
    nLineagesExtinct = sum(1 for info in lineages.values() if info['liveCount'] == 0)
    nLineagesPresentAtEnd = sum(1 for info in lineages.values() if info['liveCount'] > 0)

    # ----- Rescue-specific stats -----
    # Uses the snapshot taken at rescue trigger, for the same reason as chainModel.
    if rescued:
        nLineagesAtRescue = len(rescueRLineages)
        if rescueRLineages:
            def primaryKey(item):
                lin, cnt = item
                return (cnt, lineages[lin]['birthTime'])
            primaryLineage, primaryLineageCount = max(rescueRLineages.items(), key=primaryKey)
            primaryLineageBirthTime = lineages[primaryLineage]['birthTime']
        else:
            primaryLineage = None
            primaryLineageCount = 0
            primaryLineageBirthTime = None
    else:
        nLineagesAtRescue = None
        primaryLineage = None
        primaryLineageCount = None
        primaryLineageBirthTime = None

    return {
        'rescued': rescued,
        'rescueTime': rescueTime,
        'rescueGeneration': rescueGeneration,
        # No compartments in well-mixed; rescueMode kept in schema but None.
        'rescueMode': None,
        'extinct': extinct,
        'extinctionTime': time if extinct else None,
        'extinctionGeneration': generations if extinct else None,
        'terminationReason': terminationReason,
        'finalN': N,
        'finalTime': time,
        'finalGenerations': generations,
        'mutationEvents': mutationEvents,
        'nLineagesAppeared': nLineagesAppeared,
        'nLineagesExtinct': nLineagesExtinct,
        'nLineagesPresentAtEnd': nLineagesPresentAtEnd,
        'rAtEnd': r,
        'nLineagesAtRescue': nLineagesAtRescue,
        'rescueRLineages': rescueRLineages,
        'primaryLineage': primaryLineage,
        'primaryLineageCount': primaryLineageCount,
        'primaryLineageBirthTime': primaryLineageBirthTime,
    }