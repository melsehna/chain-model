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
    p.add_argument('--maxTime', type=float, default=200.0)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    if args.condition == 'biofilm':
        params = buildChainParams(args.dose, args.mu, args.bCoreRatio, L_BIOFILM)
        simulator = simulateChain
        isChain = True
    elif args.condition == 'chainWM':
        params = buildChainParams(args.dose, args.mu, args.bCoreRatio, L_CHAIN_WM)
        simulator = simulateChain
        isChain = True
    else:
        params = buildWellMixedParams(args.dose, args.mu)
        simulator = simulateWellMixed
        isChain = False

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed', 'condition', 'dose', 'mu', 'bCoreRatio',
            'rescued', 'rescueTime', 'rescueMode',
            'extinct', 'extinctionTime',
            'finalN', 'terminationReason',
            'nMutCore', 'nMutEdge',
        ])

        for seedOffset in range(args.seedCount):
            seed = args.seedStart + seedOffset
            r = simulator(params, seed=seed, maxTime=args.maxTime,
                          stopAtRescue=True)
            if isChain:
                nMutCore = r['nMutCore']
                nMutEdge = r['nMutEdge']
                bCoreRatioOut = args.bCoreRatio
            else:
                nMutCore = 0
                nMutEdge = r['nMut']
                bCoreRatioOut = ''  # N/A for wellMixed

            writer.writerow([
                seed, args.condition, args.dose, args.mu, bCoreRatioOut,
                int(r['rescued']),
                r['rescueTime'] if r['rescueTime'] is not None else '',
                r['rescueMode'] if r['rescueMode'] is not None else '',
                int(r['extinct']),
                r['extinctionTime'] if r['extinctionTime'] is not None else '',
                r['finalN'],
                r['terminationReason'],
                nMutCore, nMutEdge,
            ])

    print(f'Wrote {args.seedCount} rows to {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()