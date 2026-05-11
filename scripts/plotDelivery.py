'''Delivery + bolus plots for the §7 (mutation fate) section.

Two side-by-side panels, both vs dose, both biofilm-only:
    A: delivery fraction = (core-born R lineages that ever reached edge) /
       (core-born R lineages that appeared). Rises with dose.
    B: mean delivery size = mean lineage size at first edge entry, for
       delivered core-born lineages. Rises with dose. The bolus signature.

For each panel we show two b_c values (moderate 0.2, active 0.8) to make
the point that the qualitative shape is structural, not specific to one
b_c.
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))


# ---- style (matches other plots) ----------------------------------------
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

STYLES = {
    'bf_mod': dict(
        color='#000000', linestyle='--', marker='s',
        markersize=11, markerfacecolor='#bfbfbf', markeredgewidth=2,
        linewidth=2,
        label=r'biofilm, $b_c = 0.2$',
    ),
    'bf_active': dict(
        color='#000000', linestyle='-', marker='^',
        markersize=13, markerfacecolor='#000000', markeredgewidth=2,
        linewidth=2.5,
        label=r'biofilm, $b_c = 0.8$',
    ),
}


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def _delivery_fraction(df):
    '''Aggregate delivery fraction: total delivered core lineages / total
    core lineages, per dose.'''
    g = df.groupby('dose').agg(
        nDel=('nDelivered', 'sum'),
        nCore=('nMutCore', 'sum'),
    ).reset_index()
    g['p'] = np.where(g.nCore > 0, g.nDel / g.nCore.replace(0, np.nan), np.nan)
    return g.sort_values('dose')


def _mean_bolus_size(df):
    '''Mean over runs of meanDeliverySizeCore (the run-level mean of lineage
    size at first edge entry for delivered core-born lineages).'''
    sub = df.copy()
    sub['meanDeliverySizeCore'] = pd.to_numeric(sub['meanDeliverySizeCore'],
                                                 errors='coerce')
    g = sub.dropna(subset=['meanDeliverySizeCore']).groupby('dose').agg(
        size=('meanDeliverySizeCore', 'mean'),
        n=('meanDeliverySizeCore', 'count'),
    ).reset_index()
    return g.sort_values('dose')


def plot_delivery(main_df, sc_df, outPath):
    bf_mod    = main_df[main_df.condition == 'biofilm']
    bf_active = sc_df[sc_df.bCoreRatio == 0.8]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel A: delivery fraction.
    axL = axes[0]
    for df, key in [(bf_mod, 'bf_mod'), (bf_active, 'bf_active')]:
        g = _delivery_fraction(df)
        axL.plot(g.dose, g.p, **STYLES[key])
    axL.set_xlabel(r'dose $d_e$')
    axL.set_ylabel('fraction of core mutations\nthat reach the edge')
    axL.set_ylim(0, 1)
    axL.legend(frameon=False, loc='lower right')
    axL.annotate('A', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold', ha='left', va='top')
    _style_axes(axL)

    # Panel B: bolus size.
    axR = axes[1]
    for df, key in [(bf_mod, 'bf_mod'), (bf_active, 'bf_active')]:
        g = _mean_bolus_size(df)
        axR.plot(g.dose, g['size'], **STYLES[key])
    axR.axhline(1.0, color='#888888', linestyle=':', linewidth=1.5)
    axR.set_xlabel(r'dose $d_e$')
    axR.set_ylabel('mean delivery size\n(cells on first edge entry)')
    axR.legend(frameon=False, loc='upper left')
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
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    main_df = pd.read_csv(args.mainCsv, low_memory=False)
    sc_df   = pd.read_csv(args.sensCoreCsv, low_memory=False)

    plot_delivery(main_df, sc_df,
                  os.path.join(args.outDir, 'delivery.png'))


if __name__ == '__main__':
    main()
