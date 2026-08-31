'''Run one batch of simulations for a single parameter point.

Three conditions:
    biofilm  : chainModel with l=100 (core + edge structure)
    chainWM  : chainModel with l=2000 (l >= nInit so no core forms --
               chain model reduced to its well-mixed limit)
    wellMixed: explicit well-mixed birth-death model (cross-check for chainWM)

chainWM and wellMixed should give statistically equivalent results.
Running both provides a sanity check on the chain model implementation.

Usage:
    python runBatch.py --condition biofilm   --dose 1.5 --seedStart 0 \\
                       --seedCount 200 --mu 1e-4 --bCoreRatio 0.2 --out bf.csv
    python runBatch.py --condition chainWM   --dose 1.5 --seedStart 0 \\
                       --seedCount 200 --mu 1e-4 --bCoreRatio 0.2 --out cwm.csv
    python runBatch.py --condition wellMixed --dose 1.5 --seedStart 0 \\
                       --seedCount 200 --mu 1e-4                    --out wm.csv
'''

import argparse
import csv
import sys

from chainModel import simulateChain
from wellMixed import simulateWellMixed


L_BIOFILM = 100
L_CHAIN_WM = 2000


def buildChainParams(dose, mu, bCoreRatio, lVal):
    return {
        'mu': mu,
        'nInit': 1000,
        'l': lVal,
        'bWtCore': bCoreRatio,
        'dWtCore': bCoreRatio,
        'bWtEdge': 1.0,
        'dWtEdge': dose,
        'bREdge': 1.0,
        'dREdge': 0.6,
    }


def buildWellMixedParams(dose, mu):
    return {
        'mu': mu,
        'nInit': 1000,
        'bWtEdge': 1.0,
        'dWtEdge': dose,
        'bREdge': 1.0,
        'dREdge': 0.6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', required=True,
                   choices=['biofilm', 'chainWM', 'wellMixed'])
    p.add_argument('--dose', type=float, required=True)
    p.add_argument('--seedStart', type=int, required=True)
    p.add_argument('--seedCount', type=int, required=True)
    p.add_argument('--mu', type=float, default=1e-4)
    p.add_argument('--bCoreRatio', type=float, default=0.2,
                   help='only used for chain model conditions')
    p.add_argument('--l', dest='lVal', type=int, default=None,
                   help='override edge width l for biofilm condition '
                        '(default: 100 for biofilm, 2000 for chainWM, ignored for wellMixed)')
    p.add_argument('--K', type=float, default=None,
                   help='carrying capacity for the Wilson density factor on R births '
                        '(default: nInit, i.e. current behavior). Pass a large value '
                        '(e.g. 1e9) to switch the factor off.')
    p.add_argument('--recordEvery', type=int, default=10**9,
                   help='trajectory snapshot interval. Batch runs write no trajectory to '
                        'CSV, so this defaults to effectively off; chainModel copies the '
                        'whole cell array every recordEvery events, which dominates memory '
                        'on long runs (a 350k-event run stores ~7000 snapshots at the '
                        'simulator default of 50).')
    p.add_argument('--lambdaPen', type=float, default=None,
                   help='penetration length: derive the exposed-layer width from dose as '
                        'l = lambdaPen * ln(dose/MIC) instead of using a fixed --l. '
                        'Chain conditions only; omit for the fixed-l behavior.')
    p.add_argument('--kEst', type=int, default=3,
                   help='count-based establishment criterion: lineage max live count '
                        '(default 3, error rate (dR/bR)^kEst; 8 gives ~1.7%%). The '
                        'nSurvived* columns give the exact criterion (never went extinct), '
                        'which is only unbiased in runs that are not truncated at rescue.')
    p.add_argument('--tracer', action='store_true',
                   help='neutral-marker mode: R gets the WT edge rates and rescue is '
                        'disabled, so runs go to extinction. Measures supply and delivery '
                        'without truncation bias or selection on the marker.')
    p.add_argument('--muCore', type=float, default=None,
                   help='mutation rate for core births (default: same as --mu). Set to 0 '
                        'to keep the core active but produce no core-born lineages, which '
                        'removes both the extra supply and the early-stop censoring of '
                        'edge-born lineages, leaving only the conveyor sweeping them inward.')
    p.add_argument('--costR', type=float, default=0.0,
                   help='fitness cost of resistance, expressed only where the drug is '
                        'absent: bRCore = bWtCore * (1 - costR), dRCore unchanged. 0 (default) '
                        'is the neutral-core behavior. No effect on wellMixed or on any '
                        'condition without a core.')
    p.add_argument('--densityForm', choices=['linear', 'step', 'none'], default='linear',
                   help="shape of the density factor on R births: 'linear' (Wilson, current), "
                        "'step' (K acts as a ceiling only), 'none' (no cap)")
    p.add_argument('--maxTime', type=float, default=float('inf'),
                   help='hard time cap (default inf); model terminates naturally on rescue or extinction')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if args.condition == 'biofilm':
        lVal = args.lVal if args.lVal is not None else L_BIOFILM
        params = buildChainParams(args.dose, args.mu, args.bCoreRatio, lVal)
        simulator = simulateChain
        isChain = True
    elif args.condition == 'chainWM':
        # chainWM is the well-mixed limit of the chain model; its l is fixed.
        lVal = L_CHAIN_WM
        params = buildChainParams(args.dose, args.mu, args.bCoreRatio, lVal)
        simulator = simulateChain
        isChain = True
    else:
        lVal = None
        params = buildWellMixedParams(args.dose, args.mu)
        simulator = simulateWellMixed
        isChain = False

    if args.muCore is not None:
        if not isChain:
            sys.exit('--muCore applies to the chain conditions only')
        params['muCore'] = args.muCore

    if args.costR:
        if not isChain:
            sys.exit('--costR applies to the chain conditions only (wellMixed has no core)')
        if not 0.0 <= args.costR <= 1.0:
            sys.exit(f'--costR must be in [0, 1], got {args.costR}')
        # Cost is paid only in the drug-free compartment: R births in the core are
        # slowed, R deaths are not. Edge rates (the resistant phenotype under drug)
        # are untouched, so s_R in the edge keeps its meaning.
        params['bRCore'] = params['bWtCore'] * (1.0 - args.costR)

    if args.K is not None:
        params['K'] = args.K
    params['densityForm'] = args.densityForm
    if args.lambdaPen is not None:
        if not isChain:
            sys.exit('--lambdaPen applies to the chain conditions only')
        params['lambdaPen'] = args.lambdaPen

    if args.tracer:
        # R becomes a neutral label: identical rates to WT in every compartment,
        # so the population trajectory is that of a pure-WT run and nothing selects
        # on the marker. Core R rates already default to the WT core rates.
        params['bREdge'] = params['bWtEdge']
        params['dREdge'] = params['dWtEdge']

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'condition', 'dose', 'mu', 'bCoreRatio', 'l',
            'rescued', 'rescueTime', 'rescueMode',
            'extinct', 'extinctionTime',
            'finalN', 'terminationReason',
            'nMutCore', 'nMutEdge',
            'nEstablishedEdge', 'nEstablishedCore',
            'nDelivered', 'meanDeliverySizeCore',
            'phase1EndTime',
            'nMutEdgePhase1', 'nMutEdgePhase2',
            'nEstEdgePhase1', 'nEstEdgePhase2',
            'K', 'densityForm', 'kEst', 'tracer', 'lambdaPen', 'costR', 'muCore',
            'nSurvivedEdge', 'nSurvivedCore',
            'nEdgeBornEnteredCore', 'nSweepsEdgeBorn',
        ])

        # simulateWellMixed keeps no trajectory, so it has no recordEvery argument.
        chainOnlyKw = {'recordEvery': args.recordEvery} if isChain else {}

        unexpectedTerm = 0
        for seedOffset in range(args.seedCount):
            seed = args.seedStart + seedOffset
            r = simulator(params, seed=seed, maxTime=args.maxTime,
                          stopAtRescue=not args.tracer,
                          rThreshold=float('inf') if args.tracer else 10,
                          kEst=args.kEst, **chainOnlyKw)
            if r['terminationReason'] not in ('rescue', 'extinction'):
                unexpectedTerm += 1
                print(f'WARN seed={seed} terminationReason={r["terminationReason"]} '
                      f'finalN={r["finalN"]}', file=sys.stderr)
            if isChain:
                nMutCore       = r['nLineagesAppearedCore']
                nMutEdge       = r['nLineagesAppearedEdge']
                nEstEdge       = r['nLineagesEstablishedEdge']
                nEstCore       = r['nLineagesEstablishedCore']
                nSurvEdge      = nMutEdge - r['nLineagesExtinctEdge']
                nSurvCore      = nMutCore - r['nLineagesExtinctCore']
                nEdgeBornCore  = r['nLineagesEdgeBornEnteredCore']
                nSweeps        = r['nSweepsEdgeBorn']
                nDelivered     = r['nLineagesReachedEdge']
                meanDelSize    = r['meanDeliverySizeCore']
                bCoreRatioOut  = args.bCoreRatio
                lOut           = r.get('l', lVal)
                phase1End      = r['phase1EndTime'] if r['phase1EndTime'] is not None else ''
                nMutEdgeP1     = r['nLineagesAppearedEdgePhase1']
                nMutEdgeP2     = r['nLineagesAppearedEdgePhase2']
                nEstEdgeP1     = r['nLineagesEstablishedEdgePhase1']
                nEstEdgeP2     = r['nLineagesEstablishedEdgePhase2']
            else:
                # wellMixed has no compartments and no Phase 1; treat all as Phase 2
                # for plotting consistency (panel will compare BF P1, BF P2, WM).
                nMutCore       = 0
                nMutEdge       = r['nLineagesAppeared']
                nEstEdge       = r['nLineagesEstablished']
                nEstCore       = 0
                nSurvEdge      = nMutEdge - r['nLineagesExtinct']
                nSurvCore      = 0
                nEdgeBornCore  = 0
                nSweeps        = 0
                nDelivered     = 0
                meanDelSize    = None
                bCoreRatioOut  = ''  # N/A for wellMixed
                lOut           = ''
                phase1End      = ''
                nMutEdgeP1     = 0
                nMutEdgeP2     = r['nLineagesAppeared']
                nEstEdgeP1     = 0
                nEstEdgeP2     = r['nLineagesEstablished']

            writer.writerow([
                seed, args.condition, args.dose, args.mu, bCoreRatioOut, lOut,
                int(r['rescued']),
                r['rescueTime'] if r['rescueTime'] is not None else '',
                r['rescueMode'] if r['rescueMode'] is not None else '',
                int(r['extinct']),
                r['extinctionTime'] if r['extinctionTime'] is not None else '',
                r['finalN'],
                r['terminationReason'],
                nMutCore, nMutEdge,
                nEstEdge, nEstCore,
                nDelivered,
                meanDelSize if meanDelSize is not None else '',
                phase1End,
                nMutEdgeP1, nMutEdgeP2,
                nEstEdgeP1, nEstEdgeP2,
                args.K if args.K is not None else params.get('nInit', ''),
                args.densityForm, args.kEst, int(args.tracer),
                args.lambdaPen if args.lambdaPen is not None else '',
                args.costR,
                args.muCore if args.muCore is not None else args.mu,
                nSurvEdge, nSurvCore,
                nEdgeBornCore, nSweeps,
            ])

    print(f'Wrote {args.seedCount} rows to {args.out}', file=sys.stderr)
    if unexpectedTerm:
        print(f'WARN {unexpectedTerm}/{args.seedCount} runs terminated for non-natural reason',
              file=sys.stderr)


if __name__ == '__main__':
    main()