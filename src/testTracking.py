'''Sanity checks for the revised chainModel and wellMixed.

Runs both models with small parameters and checks:
  - chainModel invariants hold under debug=True (cell-count consistency,
    per-lineage count consistency)
  - Output dicts have all the expected keys
  - Lineage accounting is internally consistent:
      appeared == extinct + present
      each lineage's birthRegion matches mutationEvents
  - wellMixed r count agrees with sum(rLineages)
  - chainModel well-mixed limit (l >= nInit) gives reasonable rescue behavior
'''

import numpy as np
from chainModel import simulateChain
from wellMixed import simulateWellMixed


EXPECTED_CHAIN_KEYS = {
    'rescued', 'rescueTime', 'rescueGeneration', 'rescueMode',
    'extinct', 'extinctionTime', 'extinctionGeneration',
    'terminationReason', 'finalN', 'finalTime', 'finalGenerations',
    'finalCells', 'mutationEvents',
    'nLineagesAppeared', 'nLineagesAppearedCore', 'nLineagesAppearedEdge',
    'nLineagesExtinctCore', 'nLineagesExtinctEdge',
    'nLineagesReachedEdge',
    'nLineagesPresentAtEnd', 'nLineagesPresentInCore', 'nLineagesPresentInEdge',
    'rCoreAtEnd', 'rEdgeAtEnd',
    'nLineagesAtRescueEdge', 'rescueEdgeCounts',
    'primaryLineage', 'primaryLineageCount',
    'primaryLineageOrigin', 'primaryLineageBirthTime',
    'trajectoryCells', 'trajectoryTimes',
    'trajectoryBoundaries', 'trajectoryGenerations',
}

EXPECTED_WM_KEYS = {
    'rescued', 'rescueTime', 'rescueGeneration', 'rescueMode',
    'extinct', 'extinctionTime', 'extinctionGeneration',
    'terminationReason', 'finalN', 'finalTime', 'finalGenerations',
    'mutationEvents',
    'nLineagesAppeared', 'nLineagesExtinct', 'nLineagesPresentAtEnd',
    'rAtEnd',
    'nLineagesAtRescue', 'rescueRLineages',
    'primaryLineage', 'primaryLineageCount',
    'primaryLineageBirthTime',
}


def checkChainConsistency(out, params, label=''):
    missing = EXPECTED_CHAIN_KEYS - set(out.keys())
    extra = set(out.keys()) - EXPECTED_CHAIN_KEYS
    assert not missing, f'{label}: missing chainModel keys: {missing}'
    # Extras are OK (forward compat), but flag them
    if extra:
        print(f'{label}: chainModel extra keys: {extra}')

    appearedCore = out['nLineagesAppearedCore']
    appearedEdge = out['nLineagesAppearedEdge']
    appeared = appearedCore + appearedEdge

    extinctCore = out['nLineagesExtinctCore']
    extinctEdge = out['nLineagesExtinctEdge']
    extinct = extinctCore + extinctEdge

    present = out['nLineagesPresentAtEnd']

    assert appeared == extinct + present, (
        f'{label}: appeared ({appeared}) != extinct ({extinct}) + present ({present})'
    )

    # mutationEvents length should equal total appeared
    assert len(out['mutationEvents']) == appeared, (
        f'{label}: len(mutationEvents) {len(out["mutationEvents"])} != appeared {appeared}'
    )

    # mutationEvents region counts should match appearedCore/appearedEdge
    fromEvents = {'core': 0, 'edge': 0}
    for ev in out['mutationEvents']:
        fromEvents[ev['region']] += 1
    assert fromEvents['core'] == appearedCore, f'{label}: event core count mismatch'
    assert fromEvents['edge'] == appearedEdge, f'{label}: event edge count mismatch'

    # nLineagesReachedEdge: only meaningful for core-born. Bounded by appearedCore.
    assert 0 <= out['nLineagesReachedEdge'] <= appearedCore, (
        f'{label}: nLineagesReachedEdge out of range'
    )

    # At termination: cells count of R should match rCoreAtEnd + rEdgeAtEnd
    finalCells = out['finalCells']
    N = out['finalN']
    l = params['l']
    bFinal = max(0, N - l)
    actualRCore = sum(1 for c in finalCells[:bFinal] if c >= 2)
    actualREdge = sum(1 for c in finalCells[bFinal:] if c >= 2)
    assert actualRCore == out['rCoreAtEnd'], (
        f'{label}: rCoreAtEnd mismatch: claimed {out["rCoreAtEnd"]}, actual {actualRCore}'
    )
    assert actualREdge == out['rEdgeAtEnd'], (
        f'{label}: rEdgeAtEnd mismatch: claimed {out["rEdgeAtEnd"]}, actual {actualREdge}'
    )

    # Rescue fields
    if out['rescued']:
        assert out['rescueTime'] is not None
        assert out['primaryLineageOrigin'] in ('core', 'edge')
        assert out['rescueMode'] == out['primaryLineageOrigin']
        assert out['primaryLineageCount'] >= 1
        assert out['nLineagesAtRescueEdge'] >= 1
    else:
        assert out['nLineagesAtRescueEdge'] is None
        assert out['primaryLineage'] is None


def checkWMConsistency(out, label=''):
    missing = EXPECTED_WM_KEYS - set(out.keys())
    extra = set(out.keys()) - EXPECTED_WM_KEYS
    assert not missing, f'{label}: missing wellMixed keys: {missing}'
    if extra:
        print(f'{label}: wellMixed extra keys: {extra}')

    appeared = out['nLineagesAppeared']
    extinct = out['nLineagesExtinct']
    present = out['nLineagesPresentAtEnd']
    assert appeared == extinct + present, (
        f'{label}: appeared ({appeared}) != extinct ({extinct}) + present ({present})'
    )
    assert len(out['mutationEvents']) == appeared, (
        f'{label}: len(mutationEvents) != appeared'
    )

    if out['rescued']:
        assert out['rescueTime'] is not None
        assert out['nLineagesAtRescue'] >= 1
        assert out['primaryLineage'] is not None
        assert out['primaryLineageCount'] >= 1


def runTest(label, params, seed, useChain=True, debug=True, stopAtRescue=True):
    print(f'--- {label} ---')
    if useChain:
        out = simulateChain(params, seed=seed, debug=debug, stopAtRescue=stopAtRescue,
                            maxGenerations=50_000)
        checkChainConsistency(out, params, label=label)
    else:
        out = simulateWellMixed(params, seed=seed, stopAtRescue=stopAtRescue,
                                maxGenerations=50_000)
        checkWMConsistency(out, label=label)
    print(f'  rescued={out["rescued"]} termReason={out["terminationReason"]}')
    if useChain:
        print(f'  appeared: core={out["nLineagesAppearedCore"]} edge={out["nLineagesAppearedEdge"]}')
        print(f'  extinct: core={out["nLineagesExtinctCore"]} edge={out["nLineagesExtinctEdge"]}')
        print(f'  reachedEdge (core-born): {out["nLineagesReachedEdge"]}')
        print(f'  presentAtEnd: {out["nLineagesPresentAtEnd"]} (core={out["nLineagesPresentInCore"]} edge={out["nLineagesPresentInEdge"]})')
        print(f'  rAtEnd: core={out["rCoreAtEnd"]} edge={out["rEdgeAtEnd"]}')
        if out['rescued']:
            print(f'  rescue: mode={out["rescueMode"]} nLinAtRescue={out["nLineagesAtRescueEdge"]} primary={out["primaryLineage"]} primCount={out["primaryLineageCount"]}')
    else:
        print(f'  appeared={out["nLineagesAppeared"]} extinct={out["nLineagesExtinct"]} present={out["nLineagesPresentAtEnd"]} rAtEnd={out["rAtEnd"]}')
        if out['rescued']:
            print(f'  rescue: nLinAtRescue={out["nLineagesAtRescue"]} primary={out["primaryLineage"]} primCount={out["primaryLineageCount"]}')
    return out


if __name__ == '__main__':
    # Basic params: small biofilm, strong-ish treatment. Mutation rate high enough
    # to get multiple mutations in short runs.
    chainParams = {
        'mu': 1e-3,
        'nInit': 200,
        'l': 50,
        'bWtCore': 0.2, 'dWtCore': 0.2,
        'bRCore':  0.2, 'dRCore':  0.2,
        'bWtEdge': 1.0, 'dWtEdge': 1.5,
        'bREdge':  1.0, 'dREdge':  0.5,
    }

    # Run several seeds to exercise different code paths
    for seed in [0, 1, 2, 3, 4]:
        runTest(f'chain seed={seed}', chainParams, seed, useChain=True, debug=True,
                stopAtRescue=True)

    # Same, without stopAtRescue, to test the "runs past rescue" path
    for seed in [0, 1]:
        runTest(f'chain seed={seed} no-stop', chainParams, seed, useChain=True,
                debug=True, stopAtRescue=False)

    # Well-mixed
    wmParams = {
        'mu': 1e-3,
        'nInit': 200,
        'bWtEdge': 1.0, 'dWtEdge': 1.5,
        'bREdge':  1.0, 'dREdge':  0.5,
    }
    for seed in [0, 1, 2, 3, 4]:
        runTest(f'wm seed={seed}', wmParams, seed, useChain=False, stopAtRescue=True)

    # wellMixed limit of chain: l >= nInit -- entire biofilm is edge
    chainWMParams = dict(chainParams)
    chainWMParams['l'] = chainWMParams['nInit']  # whole thing is edge
    for seed in [0, 1]:
        out = runTest(f'chain-WM-limit seed={seed}', chainWMParams, seed,
                      useChain=True, debug=True, stopAtRescue=True)
        # In the WM limit, nLineagesAppearedCore should be 0 (no core exists)
        assert out['nLineagesAppearedCore'] == 0, 'WM-limit should have no core mutations'
        assert out['nLineagesReachedEdge'] == 0, 'no core-born lineages'

    # Edge-width=0 doesn't make sense for treatment; but test l=N/2
    partialParams = dict(chainParams)
    partialParams['l'] = 100
    for seed in [0, 1]:
        runTest(f'chain half-edge seed={seed}', partialParams, seed, useChain=True,
                debug=True, stopAtRescue=True)

    print('\nAll consistency checks passed.')