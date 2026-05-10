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
        ])

        unexpectedTerm = 0
        for seedOffset in range(args.seedCount):
            seed = args.seedStart + seedOffset
            r = simulator(params, seed=seed, maxTime=args.maxTime,
                          stopAtRescue=True)
            if r['terminationReason'] not in ('rescue', 'extinction'):
                unexpectedTerm += 1
                print(f'WARN seed={seed} terminationReason={r["terminationReason"]} '
                      f'finalN={r["finalN"]}', file=sys.stderr)
            if isChain:
                nMutCore       = r['nLineagesAppearedCore']
                nMutEdge       = r['nLineagesAppearedEdge']
                nEstEdge       = r['nLineagesEstablishedEdge']
                nEstCore       = r['nLineagesEstablishedCore']
                nDelivered     = r['nLineagesReachedEdge']
                meanDelSize    = r['meanDeliverySizeCore']
                bCoreRatioOut  = args.bCoreRatio
                lOut           = lVal
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
            ])

    print(f'Wrote {args.seedCount} rows to {args.out}', file=sys.stderr)
    if unexpectedTerm:
        print(f'WARN {unexpectedTerm}/{args.seedCount} runs terminated for non-natural reason',
              file=sys.stderr)


if __name__ == '__main__':
    main()