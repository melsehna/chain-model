'''Double-edged-sword bar plots, one per core-turnover level.

For each b_c, side-by-side subplots compare biofilm vs well-mixed at a low
and high dose. Biofilm data is drawn from sensCore.csv (which covers
b_c in {0.0, 0.05, 0.1, 0.2, 0.4, 0.8}); the well-mixed reference comes
from main.csv (one fixed run set, b_c-independent).

Default run emits two files:
    figures/panels/double_edge_rescue_lowCore.png   (b_c = 0.0, dormant)
    figures/panels/double_edge_rescue_highCore.png  (b_c = 0.8, active)
    figures/panels/double_edge_est_lowCore.png
    figures/panels/double_edge_est_highCore.png
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))


# Min font size 28 everywhere; Gillius ADF text, stixsans math.
BASE_FONT = 28
mpl.rcParams.update({
    'font.family':       'Gillius ADF',
    'font.size':         BASE_FONT,
    'axes.titlesize':    BASE_FONT + 2,
    'axes.labelsize':    BASE_FONT,
    'xtick.labelsize':   BASE_FONT,
    'ytick.labelsize':   BASE_FONT,
    'legend.fontsize':   BASE_FONT - 2,
    'figure.titlesize':  BASE_FONT + 4,
    'mathtext.fontset':  'stixsans',
})

# Black / dark grey / light grey. WM = darker, BF = lighter (or swap).
COLOR_BF = '#3a3a3a'   # dark grey
COLOR_WM = '#bfbfbf'   # light grey
EDGE     = 'black'




def p_rescue(df):
    n = len(df)
    return float(df['rescued'].sum()) / n if n else float('nan')


def p_est(df):
    nEst = int(df['nEstablishedEdge'].sum())
    nMut = int(df['nMutEdge'].sum())
    return nEst / nMut if nMut else float('nan')



def plot_double_edge_combined(bf_low_df, bf_high_df, wm_df, metric_fn, ylabel,
                              outPath, lowDose=1.05, highDose=2.8,
                              lowCoreBc=0.0, highCoreBc=0.8):
    '''Single figure with two row-panels (A = low core, B = high core).
    Each row contains two subplots (low dose, high dose) with independent
    y-scales so the BF/WM comparison is visible at both absolute levels.
    Each row is titled with the b_c condition it represents.'''
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    rows = [
        (bf_low_df,  lowCoreBc,  'A', fr'Dormant core  ($b_c = {lowCoreBc:g}$)'),
        (bf_high_df, highCoreBc, 'B', fr'Active core  ($b_c = {highCoreBc:g}$)'),
    ]
    for row_idx, (bf_df, bc, letter, row_title) in enumerate(rows):
        bf_low  = metric_fn(bf_df[bf_df.dose == lowDose])
        bf_high = metric_fn(bf_df[bf_df.dose == highDose])
        wm_low  = metric_fn(wm_df[wm_df.dose == lowDose])
        wm_high = metric_fn(wm_df[wm_df.dose == highDose])

        for col_idx, (bf_v, wm_v, header) in enumerate([
            (bf_low,  wm_low,  f'low dose  $d_e={lowDose}$'),
            (bf_high, wm_high, f'high dose  $d_e={highDose}$'),
        ]):
            ax = axes[row_idx, col_idx]
            ax.bar([0], [bf_v], 0.6,
                   color=COLOR_BF, edgecolor=EDGE, linewidth=1.5)
            ax.bar([1], [wm_v], 0.6,
                   color=COLOR_WM, edgecolor=EDGE, linewidth=1.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['biofilm', 'well-mixed'])
            ax.set_xlabel(header)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', length=6, width=1.2)
            ax.set_xlim(-0.6, 1.6)

        axes[row_idx, 0].set_ylabel(ylabel)
        # Panel letter at the upper-left corner of the row.
        axes[row_idx, 0].annotate(
            letter,
            xy=(-0.28, 1.18), xycoords='axes fraction',
            fontsize=mpl.rcParams['font.size'] + 8, fontweight='bold',
            ha='left', va='top',
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Row titles centered above each row of subplots. Y-positions are
    # tuned to sit just above each row after tight_layout.
    fig.text(0.5, 0.965, rows[0][3],
             ha='center', va='center',
             fontsize=mpl.rcParams['font.size'] + 4, fontweight='bold')
    fig.text(0.5, 0.475, rows[1][3],
             ha='center', va='center',
             fontsize=mpl.rcParams['font.size'] + 4, fontweight='bold')

    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def plot_double_edge(bf_df, wm_df, metric_fn, ylabel, outPath,
                     lowDose=1.05, highDose=2.8):
    bf_low  = metric_fn(bf_df[bf_df.dose == lowDose])
    bf_high = metric_fn(bf_df[bf_df.dose == highDose])
    wm_low  = metric_fn(wm_df[wm_df.dose == lowDose])
    wm_high = metric_fn(wm_df[wm_df.dose == highDose])

    # Two subplots, one per dose, with independent y-scales so the BF/WM
    # comparison is visible at both the high (low dose) and low (high dose)
    # absolute levels.
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    pairs = [
        (axes[0], bf_low,  wm_low,  f'low dose  $d_e={lowDose}$'),
        (axes[1], bf_high, wm_high, f'high dose  $d_e={highDose}$'),
    ]
    for ax, bf_v, wm_v, header in pairs:
        ax.bar([0], [bf_v], 0.6,
               color=COLOR_BF, edgecolor=EDGE, linewidth=1.5)
        ax.bar([1], [wm_v], 0.6,
               color=COLOR_WM, edgecolor=EDGE, linewidth=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['biofilm', 'well-mixed'])
        ax.set_xlabel(header)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', length=6, width=1.2)
        ax.set_xlim(-0.6, 1.6)

    axes[0].set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mainCsv',     default=os.path.join(REPO_ROOT, 'figures', 'main.csv'))
    p.add_argument('--sensCoreCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensCore.csv'))
    p.add_argument('--outDir',      default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    p.add_argument('--lowDose',     type=float, default=1.05)
    p.add_argument('--highDose',    type=float, default=2.8)
    p.add_argument('--lowCore',     type=float, default=0.0,
                   help='b_c value used for the "low core turnover" plots')
    p.add_argument('--highCore',    type=float, default=0.8,
                   help='b_c value used for the "high core turnover" plots')
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    main_df = pd.read_csv(args.mainCsv, low_memory=False)
    sc_df   = pd.read_csv(args.sensCoreCsv, low_memory=False)
    wm_df   = main_df[main_df.condition == 'wellMixed']

    bf_low_df  = sc_df[sc_df.bCoreRatio == args.lowCore]
    bf_high_df = sc_df[sc_df.bCoreRatio == args.highCore]
    for label, sub in [('lowCore', bf_low_df), ('highCore', bf_high_df)]:
        if sub.empty:
            raise SystemExit(f'no rows in sensCore.csv for {label} bCoreRatio')

    # Separate single-row plots (kept for back-compat / standalone use).
    for tag, bf_df in [('lowCore', bf_low_df), ('highCore', bf_high_df)]:
        plot_double_edge(
            bf_df, wm_df, p_rescue,
            ylabel=r'$P(\mathrm{rescue})$',
            outPath=os.path.join(args.outDir, f'double_edge_rescue_{tag}.png'),
            lowDose=args.lowDose, highDose=args.highDose,
        )
        plot_double_edge(
            bf_df, wm_df, p_est,
            ylabel=r'$P(\mathrm{est} \mid \mathrm{edge\ mutation})$',
            outPath=os.path.join(args.outDir, f'double_edge_est_{tag}.png'),
            lowDose=args.lowDose, highDose=args.highDose,
        )

    # Combined figure: panel A (low core) on top, panel B (high core) below.
    plot_double_edge_combined(
        bf_low_df, bf_high_df, wm_df, p_rescue,
        ylabel=r'$P(\mathrm{rescue})$',
        outPath=os.path.join(args.outDir, 'double_edge_rescue.png'),
        lowDose=args.lowDose, highDose=args.highDose,
        lowCoreBc=args.lowCore, highCoreBc=args.highCore,
    )
    plot_double_edge_combined(
        bf_low_df, bf_high_df, wm_df, p_est,
        ylabel=r'$P(\mathrm{est} \mid \mathrm{edge\ mutation})$',
        outPath=os.path.join(args.outDir, 'double_edge_est.png'),
        lowDose=args.lowDose, highDose=args.highDose,
        lowCoreBc=args.lowCore, highCoreBc=args.highCore,
    )


if __name__ == '__main__':
    main()
