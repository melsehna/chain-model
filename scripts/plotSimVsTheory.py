'''Plot simulated rescue probabilities against analytical predictions.

Loads a sweep CSV (default: results/main.csv) and overlays the analytical
curves from plotTheory.py:

  Panel 1 - P(rescue) vs dose, all 3 conditions, with theory overlay
            (Lambda upper bound and full approximation).
  Panel 2 - Pathway breakdown for biofilm rescues (core vs edge).
  Panel 3 - chainWM vs wellMixed agreement check.
  Panel 4 - Ratio P_BF/P_WM vs dose with analytical overlay.

Usage:
    python3 scripts/plotSimVsTheory.py
    python3 scripts/plotSimVsTheory.py --csv scripts/runJobs/quick_sweep.csv
    python3 scripts/plotSimVsTheory.py --csv results/main.csv \
                                       --out figures/sim_vs_theory.png
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, THIS_DIR)

from plotTheory import (
    B_C_DEF, N0, L,
    lam_wm, lam_bf,
    P_rescue_bf_approx, P_rescue_wm_approx,
)

CONDITION_COLORS = {
    'biofilm':   'C0',
    'chainWM':   'C1',
    'wellMixed': 'C2',
}


def summarize(df):
    grp = (df.groupby(['condition', 'dose'])
             .agg(n=('seed', 'count'),
                  nRescued=('rescued', 'sum'))
             .reset_index())
    grp['p'] = grp.nRescued / grp.n
    grp['se'] = np.sqrt(grp.p * (1 - grp.p) / grp.n)
    return grp


def panel_rescue_vs_dose(ax, summary, b_c):
    '''Sim points + analytical curves.'''
    doseGrid = np.linspace(1.02, 2.85, 400)
    sGrid = doseGrid - 1.0

    pwm_upper = 1 - np.exp(-lam_wm(sGrid))
    pbf_upper = 1 - np.exp(-lam_bf(sGrid, b_c))
    pwm_full = np.array([P_rescue_wm_approx(s) for s in sGrid])
    pbf_full = np.array([P_rescue_bf_approx(s, b_c) for s in sGrid])

    ax.plot(doseGrid, pwm_upper, color=CONDITION_COLORS['wellMixed'],
            ls=':', lw=1.2, alpha=0.6, label='WM upper bound (Lambda)')
    ax.plot(doseGrid, pbf_upper, color=CONDITION_COLORS['biofilm'],
            ls=':', lw=1.2, alpha=0.6, label='BF upper bound (Lambda)')
    ax.plot(doseGrid, pwm_full, color=CONDITION_COLORS['wellMixed'],
            ls='--', lw=1.5, label='WM full approx')
    ax.plot(doseGrid, pbf_full, color=CONDITION_COLORS['biofilm'],
            ls='--', lw=1.5, label='BF full approx')

    for cond in ['biofilm', 'chainWM', 'wellMixed']:
        sub = summary[summary.condition == cond].sort_values('dose')
        if len(sub) == 0:
            continue
        ax.errorbar(sub.dose, sub.p, yerr=sub.se,
                    marker='o', ms=4, lw=1, capsize=2,
                    color=CONDITION_COLORS[cond], label=f'{cond} (sim)')

    ax.set_xlabel(r'Dose $(d_e)$')
    ax.set_ylabel(r'$P_{rescue}$')
    ax.set_title(rf'Rescue Probability ($b_c$={b_c})')
    ax.set_yscale('log')
    ax.set_ylim(max(1e-4, ax.get_ylim()[0]), 1.1)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3, which='both')


def panel_pathway(ax, df):
    biofilmRescued = df[(df.condition == 'biofilm') & (df.rescued == 1)]
    if len(biofilmRescued) == 0:
        ax.text(0.5, 0.5, 'no biofilm rescues yet',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Rescue Pathway')
        return
    pathway = (biofilmRescued.groupby('dose').rescueMode
               .value_counts(normalize=True)
               .unstack(fill_value=0).sort_index())
    if 'core' in pathway.columns:
        ax.plot(pathway.index, pathway['core'], 'o-',
                label='Core Delivery', color='C3')
    if 'edge' in pathway.columns:
        ax.plot(pathway.index, pathway['edge'], 'o-',
                label='Edge Mutation', color='C4')
    ax.set_xlabel(r'Dose $(d_e)$')
    ax.set_ylabel('Fraction of Biofilm Rescues')
    ax.set_title('Pathway Breakdown')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def panel_agreement(ax, summary):
    cwm = summary[summary.condition == 'chainWM'].sort_values('dose')
    wm = summary[summary.condition == 'wellMixed'].sort_values('dose')
    if len(cwm) == 0 or len(wm) == 0:
        ax.text(0.5, 0.5, 'agreement check needs chainWM and wellMixed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Agreement check')
        return
    merged = cwm[['dose', 'p', 'se']].merge(
        wm[['dose', 'p', 'se']], on='dose', suffixes=('_chainWM', '_wellMixed'))
    merged['diff'] = merged.p_chainWM - merged.p_wellMixed
    merged['diffSE'] = np.sqrt(merged.se_chainWM**2 + merged.se_wellMixed**2)
    ax.errorbar(merged.dose, merged['diff'], yerr=merged.diffSE,
                marker='o', capsize=3, color='black')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('dose (d_e)')
    ax.set_ylabel('P(chainWM) - P(wellMixed)')
    ax.set_title('Agreement check\n(should scatter around 0)')
    ax.grid(alpha=0.3)


def panel_ratio(ax, summary, b_c):
    '''Simulated P_BF/P_WM ratio with analytical overlay.'''
    doseGrid = np.linspace(1.02, 2.85, 400)
    sGrid = doseGrid - 1.0
    pwm_full = np.array([P_rescue_wm_approx(s) for s in sGrid])
    pbf_full = np.array([P_rescue_bf_approx(s, b_c) for s in sGrid])
    ratio_th = pbf_full / pwm_full
    ax.semilogy(doseGrid, ratio_th, 'k--', lw=1.5,
                label='analytical (full approx)')

    bf = summary[summary.condition == 'biofilm'].sort_values('dose')
    wm = summary[summary.condition == 'wellMixed'].sort_values('dose')
    if len(bf) == 0 or len(wm) == 0:
        ax.set_title('Ratio P_BF / P_WM (no sim points)')
        ax.legend(fontsize=8)
        return
    merged = bf[['dose', 'p', 'se', 'n']].merge(
        wm[['dose', 'p', 'se', 'n']], on='dose', suffixes=('_bf', '_wm'))
    valid = (merged.p_bf > 0) & (merged.p_wm > 0)
    m = merged[valid]
    ratio_sim = m.p_bf / m.p_wm
    log_se = np.sqrt((m.se_bf / m.p_bf)**2 + (m.se_wm / m.p_wm)**2)
    lo = ratio_sim * np.exp(-log_se)
    hi = ratio_sim * np.exp(+log_se)
    ax.errorbar(m.dose, ratio_sim, yerr=[ratio_sim - lo, hi - ratio_sim],
                marker='o', ms=5, capsize=3, ls='',
                color='C5', label='simulation')
    ax.axhline(1.0, color='gray', ls=':', lw=1.0)
    ax.set_xlabel('dose (d_e)')
    ax.set_ylabel('P_BF / P_WM')
    ax.set_title(f'Rescue ratio: sim vs theory  (b_c={b_c})')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, which='both')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default=os.path.join(REPO_ROOT, 'results', 'main.csv'))
    p.add_argument('--out', default=os.path.join(REPO_ROOT, 'figures', 'sim_vs_theory.png'))
    p.add_argument('--bCoreRatio', type=float, default=B_C_DEF,
                   help='b_c value for analytical overlay (default: project default)')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f'CSV not found: {args.csv}')

    df = pd.read_csv(args.csv)
    print(f'Loaded {len(df):,} rows from {args.csv}')
    print(df.groupby('condition').size().to_string())

    summary = summarize(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panel_rescue_vs_dose(axes[0, 0], summary, args.bCoreRatio)
    panel_ratio(axes[0, 1], summary, args.bCoreRatio)
    panel_pathway(axes[1, 0], df)
    panel_agreement(axes[1, 1], summary)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
