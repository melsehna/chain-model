'''b_c-sensitivity plots: rescue probability across dose for biofilms
at multiple core-activity levels, plus the analogous crossover-dose
vs b_c summary. Mirrors plotSensLCurves.py.

Two outputs:
    figures/panels/sensCore_curves.png   --- 2 panels:
        A: P(rescue) vs dose at every b_c + well-mixed reference
        B: P_BF - P_WM vs dose (delta plot; zero crossing = crossover)
    figures/panels/crossover_vs_bc.png   --- crossover dose as a function
        of b_c, extracted with the same strict definition as the l plot.

The data is biofilm at l = 100 (default), sweeping b_c.
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

# Gradient over the swept b_c values
_BC_LIST = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
_bc_pos = np.linspace(0.1, 0.85, len(_BC_LIST))
_GRADIENT = plt.get_cmap('plasma')
BC_COLORS = {bc: _GRADIENT(p) for bc, p in zip(_BC_LIST, _bc_pos)}

# emphasize default b_c (main sweep used 0.2) 
BC_MARKER = {0.0: 's', 0.05: 'v', 0.1: 'o', 0.2: '^', 0.4: 'D', 0.8: 'h'}
BC_LINEWIDTH = {bc: (3.5 if bc == 0.2 else 2.2) for bc in _BC_LIST}
BC_MARKERSIZE = {bc: (14 if bc == 0.2 else 11) for bc in _BC_LIST}


def _bc_style(bc):
    label = fr'$b_c = {bc:g}$' + ('  (dormant)' if bc == 0 else
                                   '  (default)' if bc == 0.2 else '')
    return dict(
        color=BC_COLORS[bc], linestyle='-', marker=BC_MARKER[bc],
        markersize=BC_MARKERSIZE[bc],
        markeredgecolor='black', markeredgewidth=1.2,
        linewidth=BC_LINEWIDTH[bc],
        label=label,
    )


WM_STYLE = dict(
    color='black', linestyle='--', marker='o',
    markersize=11, markerfacecolor='white', markeredgewidth=2.0,
    linewidth=2.5, label='well-mixed reference',
)


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def crossover_dose(bf_p_by_dose, wm_p_by_dose):
    '''Same strict definition as plotSensLCurves.crossover_dose: the dose
    above which the biofilm is above well-mixed at every higher sampled
    dose. NaN if no such dose exists in the swept range.'''
    doses = sorted(bf_p_by_dose.index)
    diff = [bf_p_by_dose[d] - wm_p_by_dose[d] for d in doses]
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


def plot_senscore(main_df, sc_df, outPath, bc_values=_BC_LIST):
    wm = (main_df[main_df.condition == 'wellMixed']
          .groupby('dose').rescued.mean().reset_index().sort_values('dose'))
    wm_by_dose = wm.set_index('dose').rescued

    fig, (ax_abs, ax_delta) = plt.subplots(1, 2, figsize=(17, 8))

    # panel a: absolute prescue
    for bc in bc_values:
        sub = (sc_df[sc_df.bCoreRatio == bc].groupby('dose').rescued.mean()
               .reset_index().sort_values('dose'))
        ax_abs.plot(sub.dose, sub.rescued, **_bc_style(bc))
    ax_abs.plot(wm.dose, wm.rescued, **WM_STYLE)
    ax_abs.set_yscale('log')
    ax_abs.set_xlabel(r'dose $d_e$')
    ax_abs.set_ylabel(r'$P(\mathrm{rescue})$')
    ax_abs.annotate('A', xy=(-0.16, 1.02), xycoords='axes fraction',
                     fontsize=mpl.rcParams['font.size'] + 8, fontweight='bold',
                     ha='left', va='top')
    _style_axes(ax_abs)

    # panel b: delta prescue
    for bc in bc_values:
        sub = (sc_df[sc_df.bCoreRatio == bc].groupby('dose').rescued.mean()
               .reset_index().sort_values('dose'))
        delta = sub.rescued.values - wm_by_dose.reindex(sub.dose).values
        ax_delta.plot(sub.dose, delta, **_bc_style(bc))
    ax_delta.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax_delta.set_xlabel(r'dose $d_e$')
    ax_delta.set_ylabel(r'$P_{\rm biofilm}(\mathrm{rescue}) - P_{\rm well\text{-}mixed}(\mathrm{rescue})$')
    ax_delta.annotate('B', xy=(-0.16, 1.02), xycoords='axes fraction',
                       fontsize=mpl.rcParams['font.size'] + 8, fontweight='bold',
                       ha='left', va='top')
    _style_axes(ax_delta)

    # legend
    handles, labels = ax_abs.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title=r'edge width $l = 100$',
        title_fontsize=mpl.rcParams['font.size'] - 2,
        frameon=False,
        loc='lower center', bbox_to_anchor=(0.5, -0.06),
        ncol=4,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def plot_crossover_vs_bc(main_df, sc_df, outPath):
    wm = main_df[main_df.condition == 'wellMixed'].groupby('dose').rescued.mean()
    bc_vals = sorted(sc_df.bCoreRatio.unique())
    xover = []
    for bc in bc_vals:
        bf = sc_df[sc_df.bCoreRatio == bc].groupby('dose').rescued.mean()
        xover.append(crossover_dose(bf, wm))

    fig, ax = plt.subplots(figsize=(11, 7))
    finite_mask = [not np.isnan(x) for x in xover]
    # connecting line through the finite points only
    ax.plot([bc for bc, m in zip(bc_vals, finite_mask) if m],
            [x for x, m in zip(xover, finite_mask) if m],
            color='black', linestyle='-', linewidth=2.5,
            marker='o', markersize=14, markerfacecolor='#000000',
            markeredgewidth=2)
    # if any b_c has no crossover, mark it with an open marker at the top
    if not all(finite_mask):
        max_swept = max(sc_df.dose.unique())
        for bc, m in zip(bc_vals, finite_mask):
            if not m:
                ax.plot([bc], [max_swept], color='black',
                        marker='o', markersize=14,
                        markerfacecolor='white', markeredgewidth=2)
        ax.text(bc_vals[0] * 1.1 if bc_vals[0] > 0 else 0.005, max_swept,
                '  no crossover in swept range',
                va='center', ha='left',
                fontsize=mpl.rcParams['font.size'] - 6, style='italic')
    ax.set_xlabel(r'core activity $b_c$')
    ax.set_ylabel('crossover dose')
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--mainCsv',     default=os.path.join(REPO_ROOT, 'figures', 'main.csv'))
    p.add_argument('--sensCoreCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensCore.csv'))
    p.add_argument('--outDir',      default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    main_df = pd.read_csv(args.mainCsv, low_memory=False)
    sc_df   = pd.read_csv(args.sensCoreCsv, low_memory=False)

    # For visual consistency, cap all curves at the same seed count.
    # bCore=0 and bCore=0.2 in sensCore have been boosted to 100k for
    # Figure 4 Panel B; the other bCore values are at 20k after the §6
    # boost. Subsetting both inputs to seed < 20000 puts every curve on
    # equal footing in this figure.
    SEED_CAP = 20000
    main_df = main_df[main_df.seed < SEED_CAP]
    sc_df   = sc_df[sc_df.seed < SEED_CAP]

    plot_senscore(main_df, sc_df,
                   os.path.join(args.outDir, 'sensCore_curves.png'))
    plot_crossover_vs_bc(main_df, sc_df,
                          os.path.join(args.outDir, 'crossover_vs_bc.png'))


if __name__ == '__main__':
    main_cli()
