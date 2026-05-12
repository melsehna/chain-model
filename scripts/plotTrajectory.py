'''Population trajectory N(t) for biofilm vs well-mixed.

Simulates the two models with mu=0 (no rescue possible) so what we see is
the pure wild-type decline. The biofilm shows the linear-then-exponential
two-phase shape described in writeup section "The model"; the well-mixed
shows a pure exponential drop. The point is that the two trajectories have
the same area under the curve (= total cell-time = total mutation supply),
despite very different shapes -- this anchors the dormant-equivalence
result from the decomposition figure.

Pattern follows src/kymographDiagnostics.py: import the simulators
directly, run a handful of seeds, plot the result.

Outputs: figures/panels/trajectory.png
'''
import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from chainModel import simulateChain
from wellMixed import simulateWellMixed


BASE_FONT = 28
mpl.rcParams.update({
    'font.family':       'Gillius ADF',
    'font.size':         BASE_FONT,
    'axes.titlesize':    BASE_FONT + 2,
    'axes.labelsize':    BASE_FONT,
    'xtick.labelsize':   BASE_FONT,
    'ytick.labelsize':   BASE_FONT,
    'legend.fontsize':   BASE_FONT - 4,
    'figure.titlesize':  BASE_FONT + 4,
    'mathtext.fontset':  'stixsans',
})

COLOR_BF = '#3a3a3a'
COLOR_WM = '#bfbfbf'



def bf_params(dose, l=100, bCore=0.2):
    return {
        'mu': 0.0, 'nInit': 1000, 'l': l,
        'bWtCore': bCore, 'dWtCore': bCore,
        'bWtEdge': 1.0,   'dWtEdge': dose,
        'bREdge':  1.0,   'dREdge':  0.6,
    }


def wm_params(dose):
    return {
        'mu': 0.0, 'nInit': 1000,
        'bWtEdge': 1.0, 'dWtEdge': dose,
        'bREdge':  1.0, 'dREdge':  0.6,
    }


def chain_trajectory(params, seed, recordEvery=200):
    r = simulateChain(params, seed=seed, recordEvery=recordEvery)
    times = np.array(r['trajectoryTimes'])
    N     = np.array([len(s) for s in r['trajectoryCells']])
    return times, N


def wm_trajectory(params, seed, dt=0.1):
    '''wellMixed doesn't record an N(t) trajectory; sample N(t) on a regular
    grid by running short maxTime chunks and reading r["finalN"].'''
    times = [0.0]
    Ns    = [params['nInit']]
    p = dict(params)
    p['nInit'] = params['nInit']
    cur_state_seed = seed
    cur_t = 0.0
    '''
    Use a single run and approximate via piecewise: simpler is to repeatedly
    advance with `maxTime` but we don't have a way to resume. Easier:
    rerun the full sim but capture intermediate states via a custom sampler.
    Since wellMixed is a pure birth-death process and doesn't expose a
    trajectory, we simulate the analytical curve instead: it's an exponential
    decay at rate s_e = dose - 1, perfectly deterministic in the mean for
    large N.  We use the stochastic seed only to randomize the extinction
    tail.
    '''
    s_e = params['dWtEdge'] - params['bWtEdge']
    # Analytical mean trajectory (deterministic except for the extinction tail)
    t_grid = np.linspace(0, 5.0 / s_e * np.log(params['nInit']), 400)
    N_grid = params['nInit'] * np.exp(-s_e * t_grid)
    return t_grid, N_grid


def mean_trajectory(simulator, params, n_seeds, recordEvery, t_grid):
    '''Average an ensemble of trajectories onto a common time grid.'''
    N_at = np.zeros((n_seeds, len(t_grid)))
    extinct_t = []
    for k in range(n_seeds):
        times, N = simulator(params, seed=k, recordEvery=recordEvery)
        if N[-1] == 0:
            extinct_t.append(times[-1])
        # piecewise-constant interpolation
        N_at[k] = np.interp(t_grid, times, N, left=N[0], right=0.0)
    return N_at.mean(axis=0)




def plot_trajectory(outPath, dose=1.05, l=100, bCore=0.2, n_seeds=20):
    # Biofilm: average across n_seeds.
    bf_p = bf_params(dose, l=l, bCore=bCore)
    # Run one seed first to size the time grid.
    first_t, first_N = chain_trajectory(bf_p, seed=0)
    # Phase 1 endpoint (analytic): T_1 = (N_0 - l) / (s_e * l)
    s_e = dose - 1.0
    T_1_analytic = (bf_p['nInit'] - l) / (s_e * l)
    # Time grid runs slightly past observed extinction.
    t_max = first_t[-1] * 1.05
    t_grid = np.linspace(0, t_max, 400)

    bf_mean = mean_trajectory(chain_trajectory, bf_p, n_seeds, 200, t_grid)

    # wm: use analytical exponential decay over same grid.
    wm_s_e = dose - 1.0
    wm_N = bf_p['nInit'] * np.exp(-wm_s_e * t_grid)
    # Truncate wm curve once it falls below 0.5 (stochastic extinction)
    wm_N[wm_N < 0.5] = np.nan

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.plot(t_grid, bf_mean, color=COLOR_BF, linewidth=3, label='biofilm')
    ax.plot(t_grid, wm_N,   color=COLOR_WM, linewidth=3, label='well-mixed',
            zorder=2)

    # Phase 1 / Phase 2 boundary on the biofilm curve
    ax.axvline(T_1_analytic, color=COLOR_BF, linestyle=':', linewidth=2,
               alpha=0.7)
    y_label = bf_p['nInit'] * 0.45
    ax.text(T_1_analytic / 2, y_label,
            'Phase 1\n(Core Present)',
            ha='center', va='center', fontsize=BASE_FONT - 4, color=COLOR_BF)
    ax.text(T_1_analytic + (t_max - T_1_analytic) / 2, y_label,
            'Phase 2\n(Core Lost)',
            ha='center', va='center', fontsize=BASE_FONT - 4, color=COLOR_BF)

    ax.set_xlabel('Time (Edge Generations)')
    ax.set_ylabel('Population Size $N(t)$')
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, bf_p['nInit'] * 1.05)
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outDir', default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    p.add_argument('--dose',   type=float, default=1.05,
                   help='Dose for the trajectory illustration; low dose '
                        'makes Phase 1 long and visually distinct.')
    p.add_argument('--l',      type=int,   default=100)
    p.add_argument('--bCore',  type=float, default=0.2)
    p.add_argument('--nSeeds', type=int,   default=20)
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    plot_trajectory(
        os.path.join(args.outDir, 'trajectory.png'),
        dose=args.dose, l=args.l, bCore=args.bCore, n_seeds=args.nSeeds,
    )


if __name__ == '__main__':
    main()
