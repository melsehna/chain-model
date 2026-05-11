'''Mechanism-scaling figures: explain *why* the crossover dose shifts
with l and b_c.

Three figures, all from the existing sensL/sensCore CSVs:

    figures/panels/supply_scaling.png    --- 2-panel
        A: mean core-born mutations per run vs dose, one curve per l
        B: same vs dose, one curve per b_c

    figures/panels/mech_grid.png         --- 2x2
        rows: supply (top), delivery fraction (bottom)
        cols: l sweep (left), b_c sweep (right)

    figures/panels/effective_supply.png  --- 2-panel
        A: mean (edge mutations + delivered core mutations) vs dose, per l
        B: same per b_c
        ("effective supply" --- mutations that actually arrive at the edge,
        either by being born there or by being delivered)

All three use the same gradient palette as plotSensLCurves and
plotSensCoreCurves so the story stays visually coherent across figures.
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))


# ---- aesthetics ---------------------------------------------------------
BASE_FONT = 28
mpl.rcParams.update({
    'font.family':       'Gillius ADF',
    'font.size':         BASE_FONT,
    'axes.titlesize':    BASE_FONT + 2,
    'axes.labelsize':    BASE_FONT,
    'xtick.labelsize':   BASE_FONT,
    'ytick.labelsize':   BASE_FONT,
    'legend.fontsize':   BASE_FONT - 6,
    'figure.titlesize':  BASE_FONT + 4,
    'mathtext.fontset':  'stixsans',
})

# Convention across the writeup: l sweeps use viridis, b_c sweeps use plasma.
# Keeping the two palettes distinct makes it obvious at a glance which
# parameter a figure is sweeping.
_GRADIENT_L  = plt.get_cmap('viridis')
_GRADIENT_BC = plt.get_cmap('plasma')

# l palette (matches plotSensLCurves.py).
_L_LIST = [25, 50, 100, 200, 400]
_log_l = np.log(_L_LIST)
_log_l_norm = (_log_l - _log_l.min()) / (_log_l.max() - _log_l.min())
L_COLORS = {l: _GRADIENT_L(0.1 + 0.75 * p)
            for l, p in zip(_L_LIST, _log_l_norm)}
L_MARKER = {25: 's', 50: 'v', 100: '^', 200: 'D', 400: 'h'}


def _l_style(l):
    return dict(
        color=L_COLORS[l], linestyle='-', marker=L_MARKER[l],
        markersize=14 if l == 100 else 11,
        markeredgecolor='black',
        markeredgewidth=1.5 if l == 100 else 1.2,
        linewidth=3.5 if l == 100 else 2.2,
        label=fr'$l = {l}$' + ('  (default)' if l == 100 else ''),
    )


# b_c palette (matches plotSensCoreCurves.py).
_BC_LIST = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
_bc_pos = np.linspace(0.1, 0.85, len(_BC_LIST))
BC_COLORS = {bc: _GRADIENT_BC(p) for bc, p in zip(_BC_LIST, _bc_pos)}
BC_MARKER = {0.0: 's', 0.05: 'v', 0.1: 'o', 0.2: '^', 0.4: 'D', 0.8: 'h'}


def _bc_style(bc):
    label = fr'$b_c = {bc:g}$' + ('  (dormant)' if bc == 0 else
                                   '  (default)' if bc == 0.2 else '')
    return dict(
        color=BC_COLORS[bc], linestyle='-', marker=BC_MARKER[bc],
        markersize=14 if bc == 0.2 else 11,
        markeredgecolor='black',
        markeredgewidth=1.5 if bc == 0.2 else 1.2,
        linewidth=3.5 if bc == 0.2 else 2.2,
        label=label,
    )


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


# ---- aggregation helpers ------------------------------------------------

def _mean_per_dose(df, col):
    return (df.groupby('dose')[col].mean()
              .reset_index().sort_values('dose'))


def _delivery_fraction(df):
    '''Aggregate delivery fraction across all runs at each dose.'''
    g = df.groupby('dose').agg(
        nDel=('nDelivered', 'sum'),
        nCore=('nMutCore', 'sum'),
    ).reset_index()
    g['p'] = np.where(g.nCore > 0,
                      g.nDel / g.nCore.replace(0, np.nan), np.nan)
    return g.sort_values('dose')


def _effective_supply(df):
    '''Mean per-run total of mutations that ever reach the edge: every
    edge-born mutation plus every core-born mutation that was delivered.'''
    sub = df.copy()
    sub['effective'] = sub['nMutEdge'] + sub['nDelivered']
    return _mean_per_dose(sub, 'effective')


# ---- figure 1: supply scaling -------------------------------------------

def plot_supply_scaling(sl_df, sc_df, outPath):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 8))

    for l in _L_LIST:
        sub = _mean_per_dose(sl_df[sl_df.l == l], 'nMutCore')
        axL.plot(sub.dose, sub.nMutCore, **_l_style(l))
    axL.set_xlabel(r'dose $d_e$')
    axL.set_ylabel('mean core mutations per run')
    axL.annotate('A', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold',
                 ha='left', va='top')
    axL.legend(frameon=False, title=r'edge width ($b_c = 0.2$)',
               title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axL)

    for bc in _BC_LIST:
        sub = _mean_per_dose(sc_df[sc_df.bCoreRatio == bc], 'nMutCore')
        axR.plot(sub.dose, sub.nMutCore, **_bc_style(bc))
    axR.set_xlabel(r'dose $d_e$')
    axR.set_ylabel('mean core mutations per run')
    axR.annotate('B', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold',
                 ha='left', va='top')
    axR.legend(frameon=False, title=r'core activity ($l = 100$)',
               title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axR)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


# ---- figure 2: 2x2 mechanism grid ---------------------------------------

def plot_mech_grid(sl_df, sc_df, outPath):
    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    (axTL, axTR), (axBL, axBR) = axes

    # Top row: supply.
    for l in _L_LIST:
        sub = _mean_per_dose(sl_df[sl_df.l == l], 'nMutCore')
        axTL.plot(sub.dose, sub.nMutCore, **_l_style(l))
    axTL.set_ylabel('mean core mutations\nper run')
    axTL.legend(frameon=False, title=r'edge width ($b_c = 0.2$)',
                title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axTL)

    for bc in _BC_LIST:
        sub = _mean_per_dose(sc_df[sc_df.bCoreRatio == bc], 'nMutCore')
        axTR.plot(sub.dose, sub.nMutCore, **_bc_style(bc))
    axTR.legend(frameon=False, title=r'core activity ($l = 100$)',
                title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axTR)

    # Bottom row: delivery fraction.
    for l in _L_LIST:
        g = _delivery_fraction(sl_df[sl_df.l == l])
        axBL.plot(g.dose, g.p, **_l_style(l))
    axBL.set_xlabel(r'dose $d_e$')
    axBL.set_ylabel('fraction of core mutations\nthat reach the edge')
    axBL.set_ylim(0, 1)
    _style_axes(axBL)

    for bc in _BC_LIST:
        if bc == 0:
            continue  # no core mutations to deliver
        g = _delivery_fraction(sc_df[sc_df.bCoreRatio == bc])
        axBR.plot(g.dose, g.p, **_bc_style(bc))
    axBR.set_xlabel(r'dose $d_e$')
    axBR.set_ylim(0, 1)
    _style_axes(axBR)

    for ax, letter in [(axTL, 'A'), (axTR, 'B'),
                       (axBL, 'C'), (axBR, 'D')]:
        ax.annotate(letter, xy=(-0.18, 1.02), xycoords='axes fraction',
                    fontsize=BASE_FONT + 8, fontweight='bold',
                    ha='left', va='top')

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


# ---- figure 3: effective supply -----------------------------------------

def plot_effective_supply(sl_df, sc_df, outPath):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 8))

    for l in _L_LIST:
        sub = _effective_supply(sl_df[sl_df.l == l])
        axL.plot(sub.dose, sub.effective, **_l_style(l))
    axL.set_xlabel(r'dose $d_e$')
    axL.set_ylabel('mean useful mutations per run\n(edge-born + delivered)')
    axL.annotate('A', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold',
                 ha='left', va='top')
    axL.legend(frameon=False, title=r'edge width ($b_c = 0.2$)',
               title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axL)

    for bc in _BC_LIST:
        sub = _effective_supply(sc_df[sc_df.bCoreRatio == bc])
        axR.plot(sub.dose, sub.effective, **_bc_style(bc))
    axR.set_xlabel(r'dose $d_e$')
    axR.set_ylabel('mean useful mutations per run\n(edge-born + delivered)')
    axR.annotate('B', xy=(-0.18, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold',
                 ha='left', va='top')
    axR.legend(frameon=False, title=r'core activity ($l = 100$)',
               title_fontsize=BASE_FONT - 6, loc='upper right')
    _style_axes(axR)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument('--sensLCsv',    default=os.path.join(REPO_ROOT, 'figures', 'sensL.csv'))
    p.add_argument('--sensCoreCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensCore.csv'))
    p.add_argument('--outDir',      default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    sl_df = pd.read_csv(args.sensLCsv, low_memory=False)
    sc_df = pd.read_csv(args.sensCoreCsv, low_memory=False)

    plot_supply_scaling(sl_df, sc_df,
                        os.path.join(args.outDir, 'supply_scaling.png'))
    plot_mech_grid(sl_df, sc_df,
                   os.path.join(args.outDir, 'mech_grid.png'))
    plot_effective_supply(sl_df, sc_df,
                          os.path.join(args.outDir, 'effective_supply.png'))


if __name__ == '__main__':
    main_cli()
