'''Dose-curve plots that follow the narrative bar plot.

Two figures, same aesthetics as the bar plots (Computer Modern Sans
Serif, min font 28, black / dark-grey / light-grey palette, no error
bars):

    figures/panels/dose_curves_rescue.png
        Two-panel stacked. Panel A: absolute P(rescue) vs dose, three
        lines (biofilm dormant core, biofilm active core, well-mixed).
        Panel B: delta P(rescue) = P_BF - P_WM vs dose, two lines
        (biofilm dormant, biofilm active), with a horizontal reference
        line at zero.

    figures/panels/decomposition.png
        Two side-by-side panels, ratio / delta only (absolute values
        not shown). Panel A: supply ratio S_BF / S_WM vs dose, log y.
        Panel B: delta per-mutation establishment P_est_BF - P_est_WM
        vs dose, linear y. Both panels show two lines (biofilm dormant,
        biofilm active) with a reference line at the no-difference
        value (y=1 for the ratio, y=0 for the delta).
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


def _delta_by_dose(bf_df, wm_df, fn):
    '''Aligned series of fn(bf) - fn(wm) at each shared dose.'''
    bf_g = _by_dose(bf_df, fn)
    wm_g = _by_dose(wm_df, fn)
    m = pd.merge(bf_g, wm_g, on='dose', suffixes=('_bf', '_wm'))
    m = m.sort_values('dose')
    return m['dose'].values, (m['value_bf'] - m['value_wm']).values


def _ratio_by_dose(bf_df, wm_df, fn):
    '''Aligned series of fn(bf) / fn(wm) at each shared dose.'''
    bf_g = _by_dose(bf_df, fn)
    wm_g = _by_dose(wm_df, fn)
    m = pd.merge(bf_g, wm_g, on='dose', suffixes=('_bf', '_wm'))
    m = m.sort_values('dose')
    return m['dose'].values, (m['value_bf'] / m['value_wm']).values


def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)


def _panel_label(ax, letter, dx=-0.13):
    ax.annotate(letter, xy=(dx, 1.02), xycoords='axes fraction',
                fontsize=BASE_FONT + 8, fontweight='bold',
                ha='left', va='top')


def plot_dose_curves_rescue(wm_df, bf_dormant_df, bf_active_df, outPath):
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={'height_ratios': [1.15, 1]})

    # Panel A: absolute P(rescue).
    axA = axes[0]
    for df, key in [(wm_df, 'wm'),
                    (bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        g = _by_dose(df, p_rescue)
        axA.plot(g.dose, g.value, **STYLES[key])
    axA.set_yscale('log')
    axA.set_ylabel(r'$P(\mathrm{rescue})$')
    axA.legend(frameon=False, loc='lower left')
    _panel_label(axA, 'A')
    _style_axes(axA)

    # Panel B: delta P(rescue).
    axB = axes[1]
    for df, key in [(bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        dose, delta = _delta_by_dose(df, wm_df, p_rescue)
        axB.plot(dose, delta, **STYLES[key])
    axB.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    axB.set_xlabel(r'dose $d_e$')
    axB.set_ylabel(r'$\Delta P(\mathrm{rescue})$')
    _panel_label(axB, 'B')
    _style_axes(axB)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'wrote {outPath}')


def plot_decomposition(wm_df, bf_dormant_df, bf_active_df, outPath):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    # Panel A: delta of mutation supply, symlog y to span both the
    # near-zero dormant line and the large positive active line.
    axL = axes[0]
    for df, key in [(bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        dose, delta = _delta_by_dose(df, wm_df, total_mutations)
        axL.plot(dose, delta, **STYLES[key])
    axL.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    axL.set_yscale('symlog', linthresh=0.05)
    axL.set_xlabel(r'dose $d_e$')
    axL.set_ylabel(r'$\Delta M = M_{\mathrm{BF}} - M_{\mathrm{WM}}$')
    axL.legend(frameon=False, loc='upper right')
    _panel_label(axL, 'A', dx=-0.20)
    _style_axes(axL)

    # Panel B: delta per-mutation establishment, linear y.
    axR = axes[1]
    for df, key in [(bf_dormant_df, 'bf_dormant'),
                    (bf_active_df, 'bf_active')]:
        dose, delta = _delta_by_dose(df, wm_df, p_est)
        axR.plot(dose, delta, **STYLES[key])
    axR.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    axR.set_xlabel(r'dose $d_e$')
    axR.set_ylabel(r'$\Delta P(\mathrm{est} \mid \mathrm{edge\ mut})$')
    _panel_label(axR, 'B', dx=-0.20)
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
    p.add_argument('--highCore',    type=float, default=0.2)
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
