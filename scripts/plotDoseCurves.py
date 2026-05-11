'''Dose-curve plots that follow the narrative bar plot.

Two figures, same aesthetics as the bar plots (Computer Modern Sans
Serif, min font 28, black / dark-grey / light-grey palette, no error
bars):

    figures/panels/dose_curves_rescue.png
        P(rescue) vs dose. Three lines: biofilm with dormant core,
        biofilm with active core, well-mixed reference. Generalizes the
        four-corner bar plot to the full swept dose range.

    figures/panels/decomposition.png
        Two-panel decomposition of P(rescue):
            (left)  mean total R lineages per run vs dose
            (right) per-mutation establishment fraction vs dose
        Same three conditions.
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))


# ---- style (matches plotDoubleEdge.py) ----------------------------------
BASE_FONT = 28
mpl.rcParams.update({
    'font.family':         'Gillius ADF',
    'font.size':           BASE_FONT,
    'axes.titlesize':      BASE_FONT + 2,
    'axes.labelsize':      BASE_FONT,
    'xtick.labelsize':     BASE_FONT,
    'ytick.labelsize':     BASE_FONT,
    'legend.fontsize':     BASE_FONT - 4,
    'figure.titlesize':    BASE_FONT + 4,
    'mathtext.fontset':    'stixsans',
})

# Three line styles distinguishable in greyscale: linestyle + marker + shade.
STYLES = {
    'wm': dict(
        color='#000000', linestyle='-',  marker='o',
        markersize=11, markerfacecolor='#ffffff', markeredgewidth=2,
        linewidth=2,   label='well-mixed',
    ),
    'bf_dormant': dict(
        color='#000000', linestyle='--', marker='s',
        markersize=11, markerfacecolor='#bfbfbf', markeredgewidth=2,
        linewidth=2,   label='biofilm, dormant core',
    ),
    'bf_active': dict(
        color='#000000', linestyle='-',  marker='^',
        markersize=13, markerfacecolor='#000000', markeredgewidth=2,
        linewidth=2.5, label='biofilm, active core',
    ),
}


def _by_dose(df, fn):
    g = df.groupby('dose').apply(fn).reset_index()
    g.columns = ['dose', 'value']
    return g.sort_values('dose')


def p_rescue(df):
    return float(df['rescued'].mean()) if len(df) else float('nan')


def total_mutations(df):
    return float((df['nMutCore'].fillna(0) + df['nMutEdge'].fillna(0)).mean())


def p_est(df):
    nMut = float(df['nMutEdge'].sum())
    nEst = float(df['nEstablishedEdge'].sum())
    return nEst / nMut if nMut > 0 else float('nan')


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def plot_dose_curves_rescue(wm_df, bf_dormant_df, bf_active_df, outPath):
    fig, ax = plt.subplots(figsize=(12, 7.5))

    for df, key in [(wm_df, 'wm'),
                    (bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        g = _by_dose(df, p_rescue)
        ax.plot(g.dose, g.value, **STYLES[key])

    ax.set_yscale('log')
    ax.set_xlabel(r'dose $d_e$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.legend(frameon=False, loc='lower left')
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def plot_decomposition(wm_df, bf_dormant_df, bf_active_df, outPath):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel A: mutation supply.
    axL = axes[0]
    for df, key in [(wm_df, 'wm'),
                    (bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        g = _by_dose(df, total_mutations)
        axL.plot(g.dose, g.value, **STYLES[key])
    axL.set_yscale('log')
    axL.set_xlabel(r'dose $d_e$')
    axL.set_ylabel('mean R mutations per run')
    axL.legend(frameon=False, loc='upper right')
    axL.annotate('A', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold', ha='left', va='top')
    _style_axes(axL)

    # Panel B: per-mutation establishment.
    axR = axes[1]
    for df, key in [(wm_df, 'wm'),
                    (bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        g = _by_dose(df, p_est)
        # _by_dose with p_est can drop dose values where nMut=0 because
        # of the apply pattern; reindex to keep a continuous line.
        axR.plot(g.dose, g.value, **STYLES[key])
    axR.set_xlabel(r'dose $d_e$')
    axR.set_ylabel(r'$P(\mathrm{est} \mid \mathrm{edge\ mutation})$')
    axR.annotate('B', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold', ha='left', va='top')
    _style_axes(axR)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mainCsv',     default=os.path.join(REPO_ROOT, 'figures', 'main.csv'))
    p.add_argument('--sensCoreCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensCore.csv'))
    p.add_argument('--outDir',      default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    p.add_argument('--lowCore',     type=float, default=0.0)
    p.add_argument('--highCore',    type=float, default=0.8)
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    main_df = pd.read_csv(args.mainCsv, low_memory=False)
    sc_df   = pd.read_csv(args.sensCoreCsv, low_memory=False)

    wm_df         = main_df[main_df.condition == 'wellMixed']
    bf_dormant_df = sc_df[sc_df.bCoreRatio == args.lowCore]
    bf_active_df  = sc_df[sc_df.bCoreRatio == args.highCore]

    for label, sub in [('wm', wm_df), ('bf_dormant', bf_dormant_df),
                       ('bf_active', bf_active_df)]:
        if sub.empty:
            raise SystemExit(f'no rows for {label}')

    plot_dose_curves_rescue(
        wm_df, bf_dormant_df, bf_active_df,
        os.path.join(args.outDir, 'dose_curves_rescue.png'),
    )
    plot_decomposition(
        wm_df, bf_dormant_df, bf_active_df,
        os.path.join(args.outDir, 'decomposition.png'),
    )


if __name__ == '__main__':
    main()
