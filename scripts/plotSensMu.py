'''mu-sensitivity plots: rescue probability across dose for biofilms
at multiple mutation rates, matched against the well-mixed reference.
Mirrors plotSensCoreCurves.py / plotSensLCurves.py structure.

Output:
    figures/panels/sensMu_curves.png   --- 2 panels:
        A: P(rescue) vs dose at every mu for biofilm + well-mixed reference
        B: P_BF - P_WM vs dose (delta plot; zero crossing = crossover)

The data is biofilm at l = 100, bCoreRatio = 0.2 (the defaults), sweeping mu.
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

_MU_LIST = [1e-5, 1e-4, 1e-3]
_mu_pos = np.linspace(0.15, 0.85, len(_MU_LIST))
_GRADIENT = plt.get_cmap('viridis')
MU_COLORS = {mu: _GRADIENT(p) for mu, p in zip(_MU_LIST, _mu_pos)}

MU_MARKER = {1e-5: 's', 1e-4: '^', 1e-3: 'D'}
MU_LINEWIDTH = {mu: (3.5 if mu == 1e-4 else 2.2) for mu in _MU_LIST}
MU_MARKERSIZE = {mu: (14 if mu == 1e-4 else 11) for mu in _MU_LIST}


def _mu_style(mu):
    exponent = int(np.round(np.log10(mu)))
    label = fr'$\mu = 10^{{{exponent}}}$' + ('  (default)' if mu == 1e-4 else '')
    return dict(
        color=MU_COLORS[mu], linestyle='-', marker=MU_MARKER[mu],
        markersize=MU_MARKERSIZE[mu],
        markeredgecolor='black', markeredgewidth=1.2,
        linewidth=MU_LINEWIDTH[mu],
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


def _p_rescue_by_dose(df):
    g = df.groupby('dose').rescued.mean().reset_index()
    g.columns = ['dose', 'p']
    return g.sort_values('dose')


def plot_sens_mu_curves(df, outPath):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    axA, axB = axes

    # Well-mixed reference uses mu = default (1e-4) since per-dose conditions
    # match. For consistency, pick wellMixed rows at each mu and overlay one
    # well-mixed line (using mu=1e-4 as canonical).
    wm_default = df[(df.condition == 'wellMixed') & (np.isclose(df.mu, 1e-4))]
    wm_g = _p_rescue_by_dose(wm_default)

    # Panel A: P(rescue) vs dose for each mu (biofilm), plus well-mixed.
    for mu in _MU_LIST:
        bf = df[(df.condition == 'biofilm') & (np.isclose(df.mu, mu))]
        g = _p_rescue_by_dose(bf)
        axA.plot(g.dose, g.p, **_mu_style(mu))
    axA.plot(wm_g.dose, wm_g.p, **WM_STYLE)
    axA.set_yscale('log')
    axA.set_xlabel(r'dose $d_e$')
    axA.set_ylabel(r'$P(\mathrm{rescue})$')
    axA.annotate('A', xy=(-0.16, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold', ha='left', va='top')
    _style_axes(axA)

    # Panel B: delta = P_BF - P_WM vs dose, per mu.
    for mu in _MU_LIST:
        bf = df[(df.condition == 'biofilm') & (np.isclose(df.mu, mu))]
        wm = df[(df.condition == 'wellMixed') & (np.isclose(df.mu, mu))]
        bf_g = _p_rescue_by_dose(bf).set_index('dose')
        wm_g_local = _p_rescue_by_dose(wm).set_index('dose')
        delta = (bf_g.p - wm_g_local.p).reset_index()
        axB.plot(delta.dose, delta.p, **_mu_style(mu))
    axB.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    axB.set_xlabel(r'dose $d_e$')
    axB.set_ylabel(r'$P_{\mathrm{biofilm}}(\mathrm{rescue}) - P_{\mathrm{well-mixed}}(\mathrm{rescue})$')
    axB.annotate('B', xy=(-0.16, 1.02), xycoords='axes fraction',
                 fontsize=BASE_FONT + 8, fontweight='bold', ha='left', va='top')
    _style_axes(axB)

    # Combined legend at the bottom.
    handles = []
    labels = []
    for mu in _MU_LIST:
        handles.append(plt.Line2D([], [], **_mu_style(mu)))
        labels.append(_mu_style(mu)['label'])
    handles.append(plt.Line2D([], [], **WM_STYLE))
    labels.append(WM_STYLE['label'])
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sensMuCsv', default=os.path.join(REPO_ROOT, 'figures', 'sensMu.csv'))
    p.add_argument('--outDir',    default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)
    df = pd.read_csv(args.sensMuCsv, low_memory=False)

    plot_sens_mu_curves(df, os.path.join(args.outDir, 'sensMu_curves.png'))


if __name__ == '__main__':
    main()
