'''Single-run kymograph of a core-delivery rescue event, for §7.

Pattern follows src/kymographDiagnostics.py: scan seeds at illustrative
parameters until we find a clean example where a core-born R lineage
drifts in the core, gets pulled into the edge by the retreating
boundary, and takes over.

The mutation rate is bumped up from the sweep default (mu=1e-4) so a
clean example shows up quickly; the mechanism doesn't depend on mu.

Outputs: figures/panels/kymograph_core_rescue.png
'''
import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from chainModel import simulateChain
from kymograph import plotKymograph


BASE_FONT = 28
mpl.rcParams.update({
    'font.family':       'Gillius ADF',
    'font.size':         BASE_FONT,
    'axes.titlesize':    BASE_FONT + 2,
    'axes.labelsize':    BASE_FONT,
    'xtick.labelsize':   BASE_FONT - 4,
    'ytick.labelsize':   BASE_FONT - 4,
    'legend.fontsize':   BASE_FONT - 12,   # kymograph legend is verbose
    'figure.titlesize':  BASE_FONT + 4,
    'mathtext.fontset':  'stixsans',
})


def params(dose, mu, bCore=0.2):
    return {
        'mu': mu, 'nInit': 1000, 'l': 100,
        'bWtCore': bCore, 'dWtCore': bCore,
        'bWtEdge': 1.0,   'dWtEdge': dose,
        'bREdge':  1.0,   'dREdge':  0.6,
    }


def find_core_rescue(p, max_seeds=500, recordEvery=50,
                     min_rescue_time=3.0, max_rescue_time=80.0,
                     min_final_N=100):
    '''Scan seeds for a clean core-delivery rescue. Returns (result, seed).

    Uses stopAtRescue=True so each run terminates at the rescue moment;
    without that, rescued runs equilibrate at finite N and never end.'''
    for seed in range(max_seeds):
        r = simulateChain(p, seed=seed, recordEvery=recordEvery,
                          stopAtRescue=True)
        if (r['rescued']
                and r['primaryLineageOrigin'] == 'core'
                and min_rescue_time < r['rescueTime'] < max_rescue_time
                and r['finalN'] > min_final_N):
            return r, seed
    return None, None


def render(result, params_used, outPath):
    fig, ax = plt.subplots(figsize=(13, 10))
    plotKymograph(result, params_used, title=None, ax=ax)

    # Override the auto labels with cleaner copy.
    ax.set_xlabel('cell position (0 = deep interior, $N{-}1$ = outer surface)')
    ax.set_ylabel('time (edge generations)')

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p_arg = argparse.ArgumentParser()
    p_arg.add_argument('--outDir', default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    p_arg.add_argument('--dose',   type=float, default=1.5)
    p_arg.add_argument('--mu',     type=float, default=5e-4)
    p_arg.add_argument('--bCore',  type=float, default=0.2)
    p_arg.add_argument('--maxSeeds', type=int, default=500)
    args = p_arg.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    sim_params = params(dose=args.dose, mu=args.mu, bCore=args.bCore)
    result, seed = find_core_rescue(sim_params, max_seeds=args.maxSeeds)
    if result is None:
        raise SystemExit(f'no clean core-delivery rescue in {args.maxSeeds} seeds')

    print(f'found core rescue at seed={seed}, '
          f'rescueTime={result["rescueTime"]:.2f}, '
          f'primary lineage born at t={result["primaryLineageBirthTime"]:.2f}')

    render(result, sim_params,
           os.path.join(args.outDir, 'kymograph_core_rescue.png'))


if __name__ == '__main__':
    main()
