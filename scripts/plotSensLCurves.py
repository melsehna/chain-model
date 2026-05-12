'''l-sensitivity dose curves, in the same aesthetic as the other writeup
figures. Plots P(rescue) vs dose for biofilm at multiple edge widths
(l) plus the well-mixed reference. Demonstrates that the qualitative
double-edge pattern persists across biofilm geometries, while the
magnitude of the high-dose advantage depends on l.

The sensL sweep uses biofilm at b_c = 0.2 (the default).
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


_L_LIST = [25, 50, 100, 200, 400]
_log_l = np.log(_L_LIST)
_log_l_norm = (_log_l - _log_l.min()) / (_log_l.max() - _log_l.min())
# Squeeze cmap range 
_GRADIENT = plt.get_cmap('viridis')
L_COLORS = {l: _GRADIENT(0.1 + 0.75 * p)
            for l, p in zip(_L_LIST, _log_l_norm)}

L_STYLES = {
    25: dict(
        color=L_COLORS[25],  linestyle='-', marker='s',
        markersize=11, markeredgecolor='black', markeredgewidth=1.2,
        linewidth=2.2, label=r'$l = 25$',
    ),
    50: dict(
        color=L_COLORS[50],  linestyle='-', marker='v',
        markersize=11, markeredgecolor='black', markeredgewidth=1.2,
        linewidth=2.2, label=r'$l = 50$',
    ),
    100: dict(
        color=L_COLORS[100], linestyle='-', marker='^',
        markersize=14, markeredgecolor='black', markeredgewidth=1.5,
        linewidth=3.5, label=r'$l = 100$  (default)',
    ),
    200: dict(
        color=L_COLORS[200], linestyle='-', marker='D',
        markersize=11, markeredgecolor='black', markeredgewidth=1.2,
        linewidth=2.2, label=r'$l = 200$',
    ),
    400: dict(
        color=L_COLORS[400], linestyle='-', marker='h',
        markersize=12, markeredgecolor='black', markeredgewidth=1.2,
        linewidth=2.2, label=r'$l = 400$',
    ),
}
WM_STYLE = dict(
    color='black', linestyle='--', marker='o',
    markersize=11, markerfacecolor='white', markeredgewidth=2.0,
    linewidth=2.5, label='well-mixed reference',
)


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def plot_sensl(main_df, sl_df, outPath,
               l_values=(25, 50, 100, 200, 400)):
    '''Two-panel figure: absolute P(rescue) vs dose (left) and the
    biofilm minus well-mixed difference (right). The right panel makes
    the crossover --- the dose at which the biofilm overtakes
    well-mixed --- visually direct: it is just the zero crossing.'''
    wm = (main_df[main_df.condition == 'wellMixed']
          .groupby('dose').rescued.mean().reset_index().sort_values('dose'))
    wm_by_dose = wm.set_index('dose').rescued

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(17, 8))

    # panel A: absolute P(rescue)
    for l in l_values:
        sub = (sl_df[sl_df.l == l].groupby('dose').rescued.mean()
               .reset_index().sort_values('dose'))
        ax_abs.plot(sub.dose, sub.rescued, **L_STYLES[l])
    ax_abs.plot(wm.dose, wm.rescued, **WM_STYLE)
    ax_abs.set_yscale('log')
    ax_abs.set_xlabel(r'dose $d_e$')
    ax_abs.set_ylabel(r'$P(\mathrm{rescue})$')
    ax_abs.annotate('A', xy=(-0.16, 1.02), xycoords='axes fraction',
                     fontsize=mpl.rcParams['font.size'] + 8, fontweight='bold',
                     ha='left', va='top')
    _style_axes(ax_abs)

    # panel b: delta P(rescue)
    for l in l_values:
        sub = (sl_df[sl_df.l == l].groupby('dose').rescued.mean()
               .reset_index().sort_values('dose'))
        delta = sub.rescued.values - wm_by_dose.reindex(sub.dose).values
        style = dict(L_STYLES[l])
        ax_delta.plot(sub.dose, delta, **style)
    ax_delta.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax_delta.set_xlabel(r'dose $d_e$')
    ax_delta.set_ylabel(r'$P_{\rm biofilm}(\mathrm{rescue}) - P_{\rm well\text{-}mixed}(\mathrm{rescue})$')
    ax_delta.annotate('B', xy=(-0.16, 1.02), xycoords='axes fraction',
                       fontsize=mpl.rcParams['font.size'] + 8, fontweight='bold',
                       ha='left', va='top')
    _style_axes(ax_delta)

    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title=r'active core ($b_c = 0.2$)',
        title_fontsize=mpl.rcParams['font.size'] - 2,
        frameon=False,
        loc='lower center', bbox_to_anchor=(0.5, -0.04),
        ncol=3,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def crossover_dose(bf_p_by_dose, wm_p_by_dose):
    '''Dose at which the biofilm curve crosses above the well-mixed
    reference AND stays above for every higher sampled dose.

    Strict definition: find the largest sampled dose at which the biofilm
    is still at or below well-mixed. The crossover lies between that dose
    and the next sampled dose (where the biofilm has gone above). Linearly
    interpolate $P_{\rm BF} - P_{\rm WM}$ to find the zero crossing.

    Returns NaN if the biofilm never crosses above (i.e. it stays below
    well-mixed at the highest sampled dose) or if it is above at every
    sampled dose.
    '''
    doses = sorted(bf_p_by_dose.index)
    diff = [bf_p_by_dose[d] - wm_p_by_dose[d] for d in doses]
    # last sampled dose at which the Prescue(biofilm) <= well-mixed.
    last_below = None
    for i in range(len(doses) - 1, -1, -1):
        if diff[i] <= 0:
            last_below = i
            break
    if last_below is None or last_below == len(doses) - 1:
        return float('nan')
    d_lo, d_hi = doses[last_below], doses[last_below + 1]
    y_lo, y_hi = diff[last_below], diff[last_below + 1]
    frac = -y_lo / (y_hi - y_lo)
    return d_lo + frac * (d_hi - d_lo)


def plot_crossover_vs_l(main_df, sl_df, outPath):
    '''Crossover dose (the dose at which the biofilm starts winning) as a
    function of edge width l. Computed from the same simulation data as
    plot_sensl, by linear interpolation of log(BF/WM).'''
    wm = main_df[main_df.condition == 'wellMixed'].groupby('dose').rescued.mean()
    l_vals = sorted(sl_df.l.unique())
    xover = []
    for l in l_vals:
        bf = sl_df[sl_df.l == l].groupby('dose').rescued.mean()
        xover.append(crossover_dose(bf, wm))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(l_vals, xover, color='#000000', linestyle='-',
            marker='o', markersize=14, markerfacecolor='#000000',
            markeredgewidth=2, linewidth=2.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'edge width $l$  (cells)')
    ax.set_ylabel('crossover dose')
    ax.set_xticks(l_vals)
    ax.set_xticklabels([str(int(l)) for l in l_vals])
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--mainCsv',  default=os.path.join(REPO_ROOT, 'figures', 'main.csv'))
    p.add_argument('--sensLCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensL.csv'))
    p.add_argument('--outDir',   default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    main_df = pd.read_csv(args.mainCsv, low_memory=False)
    sl_df   = pd.read_csv(args.sensLCsv, low_memory=False)

    plot_sensl(main_df, sl_df, os.path.join(args.outDir, 'sensL_curves.png'))
    plot_crossover_vs_l(main_df, sl_df,
                         os.path.join(args.outDir, 'crossover_vs_l.png'))


if __name__ == '__main__':
    main_cli()
