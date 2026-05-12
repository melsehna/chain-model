'''Phase 1 vs Phase 2 per-mutation establishment, in the same aesthetic
as plotDoubleEdge.py / plotDoseCurves.py.

Edge-born R mutations are split by birth time relative to phase1EndTime
(the moment the core empties). Phase 1 = born while the core is still
present (WT-resupply is active). Phase 2 = born after the core is gone
(WT-resupply has stopped).

Outputs:
    figures/panels/estab_phase_lines.png  (line plot across full dose range)
    figures/panels/estab_phase_bars.png   (bar plot at two doses)
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))


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

# Three categories distinguishable in greyscale.
COLOR_WM  = '#000000'
COLOR_BFP1 = '#3a3a3a'
COLOR_BFP2 = '#bfbfbf'
EDGE       = 'black'

LINE_STYLES = {
    'wm':  dict(color=COLOR_WM,   linestyle='-',  marker='o',
                markersize=11, markerfacecolor='#ffffff', markeredgewidth=2,
                linewidth=2,   label='well-mixed'),
    'bfp1': dict(color=COLOR_WM,  linestyle='--', marker='s',
                markersize=11, markerfacecolor=COLOR_BFP1, markeredgewidth=2,
                linewidth=2,   label='biofilm, Phase 1 (core present)'),
    'bfp2': dict(color=COLOR_WM,  linestyle='-',  marker='^',
                markersize=13, markerfacecolor=COLOR_BFP2, markeredgewidth=2,
                linewidth=2.5, label='biofilm, Phase 2 (core gone)'),
}


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def _by_dose(df, num_col, den_col):
    g = df.groupby('dose').agg(num=(num_col, 'sum'),
                                den=(den_col, 'sum')).reset_index()
    g['p'] = np.where(g.den > 0, g.num / g.den.replace(0, np.nan), np.nan)
    return g.sort_values('dose')


def plot_lines(main_df, outPath):
    bf = main_df[main_df.condition == 'biofilm']
    wm = main_df[main_df.condition == 'wellMixed']

    g_wm  = _by_dose(wm, 'nEstablishedEdge', 'nMutEdge')
    g_bf1 = _by_dose(bf, 'nEstEdgePhase1',   'nMutEdgePhase1')
    g_bf2 = _by_dose(bf, 'nEstEdgePhase2',   'nMutEdgePhase2')

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.plot(g_wm.dose,  g_wm.p,  **LINE_STYLES['wm'])
    ax.plot(g_bf1.dose, g_bf1.p, **LINE_STYLES['bfp1'])
    ax.plot(g_bf2.dose, g_bf2.p, **LINE_STYLES['bfp2'])

    ax.set_xlabel(r'dose $d_e$')
    ax.set_ylabel(r'$P(\mathrm{est} \mid \mathrm{edge\ mutation})$')
    ax.legend(frameon=False, loc='lower right')
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def plot_bars(main_df, outPath, lowDose=1.05, highDose=1.4):
    '''Three-bar groups (WM, BF P1, BF P2) at two doses.'''
    bf = main_df[main_df.condition == 'biofilm']
    wm = main_df[main_df.condition == 'wellMixed']

    def _vals_at(dose):
        wm_sub = wm[wm.dose == dose]
        bf_sub = bf[bf.dose == dose]
        wm_p = wm_sub.nEstablishedEdge.sum() / max(1, wm_sub.nMutEdge.sum())
        bf_p1 = bf_sub.nEstEdgePhase1.sum() / max(1, bf_sub.nMutEdgePhase1.sum())
        bf_p2 = bf_sub.nEstEdgePhase2.sum() / max(1, bf_sub.nMutEdgePhase2.sum())
        return wm_p, bf_p1, bf_p2

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, dose, header in [
        (axes[0], lowDose,  f'low dose  $d_e={lowDose}$'),
        (axes[1], highDose, f'moderate dose  $d_e={highDose}$'),
    ]:
        wm_p, bf_p1, bf_p2 = _vals_at(dose)
        xs = [0, 1, 2]
        heights = [wm_p, bf_p1, bf_p2]
        colors  = [COLOR_BFP2, COLOR_BFP1, COLOR_WM]  # light, mid, dark
        ax.bar(xs[0], heights[0], 0.6,
               color=COLOR_BFP2, edgecolor=EDGE, linewidth=1.5)
        ax.bar(xs[1], heights[1], 0.6,
               color=COLOR_BFP1, edgecolor=EDGE, linewidth=1.5)
        ax.bar(xs[2], heights[2], 0.6,
               color=COLOR_WM,   edgecolor=EDGE, linewidth=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(['well-mixed', 'BF\nPhase 1', 'BF\nPhase 2'],
                           fontsize=BASE_FONT - 2)
        ax.set_xlabel(header)
        _style_axes(ax)
        ax.set_xlim(-0.6, 2.6)

    axes[0].set_ylabel(r'$P(\mathrm{est} \mid \mathrm{edge\ mutation})$')

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mainCsv', default=os.path.join(REPO_ROOT, 'figures', 'main.csv'))
    p.add_argument('--outDir',  default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    p.add_argument('--lowDose',  type=float, default=1.05)
    p.add_argument('--highDose', type=float, default=1.4)
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    main_df = pd.read_csv(args.mainCsv, low_memory=False)

    plot_lines(main_df, os.path.join(args.outDir, 'estab_phase_lines.png'))
    plot_bars(main_df,  os.path.join(args.outDir, 'estab_phase_bars.png'),
              lowDose=args.lowDose, highDose=args.highDose)


if __name__ == '__main__':
    main()
