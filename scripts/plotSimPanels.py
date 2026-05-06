'''Per-subsection sim figure panels for writeup.tex Section 2.

Loads the three sweep CSVs (results/main.csv, results/sensCore.csv,
results/sensMu.csv) and emits PNGs into figures/panels/ so they can be
inserted inline alongside the analytical theory panels.

Outputs (matched to writeup subsections):

  figures/panels/sim_rescue_curves.png    -- P(rescue) vs dose, 3 conditions,
                                             with Lambda upper bound and full-
                                             approx theory overlay (paired with
                                             sec 2.1 "Mutation supply")
  figures/panels/sim_pathway.png          -- core vs edge fraction of biofilm
                                             rescues vs dose (paired with sec 2.2
                                             "Core delivery")
  figures/panels/sim_sensCore.png         -- biofilm rescue curves at three b_c
                                             values, with analytical full-approx
                                             curves overlaid (paired with sec 2.3
                                             "Establishment suppression")
  figures/panels/sim_ratio.png            -- P_BF / P_WM ratio with analytical
                                             prediction (paired with sec 2.4
                                             "Putting it together")
  figures/panels/sim_sensMu.png           -- biofilm vs WM rescue curves across
                                             mu in {1e-6, 1e-5, 1e-4, 1e-3}
  figures/panels/sim_agreement.png        -- chainWM - wellMixed agreement check

Usage:
    python3 scripts/plotSimPanels.py
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
    B_C_DEF, MU,
    lam_wm, lam_bf,
    P_rescue_bf_approx, P_rescue_wm_approx,
)

CONDITION_COLORS = {
    'biofilm':   'C0',
    'chainWM':   'C1',
    'wellMixed': 'C2',
}

BC_COLORS = {0.05: 'C6', 0.1: 'C8', 0.2: 'C0'}
MU_COLORS = {1e-6: 'C5', 1e-5: 'C9', 1e-4: 'C0', 1e-3: 'C3'}


def summarize(df, groupKeys):
    g = (df.groupby(groupKeys)
           .agg(n=('seed', 'count'), nRescued=('rescued', 'sum'))
           .reset_index())
    g['p'] = g.nRescued / g.n
    g['se'] = np.sqrt(g.p * (1 - g.p) / g.n)
    return g


def panel_rescue_curves(df, outPath):
    s = summarize(df, ['condition', 'dose'])
    fig, ax = plt.subplots(figsize=(7, 5))
    doseGrid = np.linspace(s.dose.min() - 0.01, s.dose.max() + 0.01, 400)
    sGrid = doseGrid - 1.0

    ax.plot(doseGrid, 1 - np.exp(-lam_wm(sGrid)),
            color=CONDITION_COLORS['wellMixed'], ls=':', lw=1.0, alpha=0.6,
            label=r'WM upper bound $1-e^{-\Lambda_{\rm WM}}$')
    ax.plot(doseGrid, 1 - np.exp(-lam_bf(sGrid, B_C_DEF)),
            color=CONDITION_COLORS['biofilm'], ls=':', lw=1.0, alpha=0.6,
            label=r'BF upper bound $1-e^{-\Lambda_{\rm BF}}$')
    ax.plot(doseGrid, np.array([P_rescue_wm_approx(s_) for s_ in sGrid]),
            color=CONDITION_COLORS['wellMixed'], ls='--', lw=1.4,
            label='WM full approx (suppression-corrected)')
    ax.plot(doseGrid, np.array([P_rescue_bf_approx(s_, B_C_DEF) for s_ in sGrid]),
            color=CONDITION_COLORS['biofilm'], ls='--', lw=1.4,
            label='BF full approx (suppression-corrected)')

    for cond in ['biofilm', 'chainWM', 'wellMixed']:
        sub = s[s.condition == cond].sort_values('dose')
        ax.errorbar(sub.dose, sub.p, yerr=sub.se,
                    marker='o', ms=4, lw=1.2, capsize=2,
                    color=CONDITION_COLORS[cond], label=f'{cond} (sim)')

    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.0)
    ax.set_title('Simulation vs theory: P(rescue) across doses\n'
                 r'($b_c=0.2$, $\mu=10^{-4}$, 5000 seeds/point)')
    ax.legend(fontsize=7.5, loc='lower left')
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_pathway(df, outPath):
    bf = df[(df.condition == 'biofilm') & (df.rescued == 1)]
    pw = (bf.groupby('dose').rescueMode
            .value_counts(normalize=True)
            .unstack(fill_value=0).sort_index())
    counts = bf.groupby('dose').rescueMode.value_counts().unstack(fill_value=0)
    nTotal = counts.sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    if 'edge' in pw.columns:
        edge = pw['edge'].reindex(nTotal.index).fillna(0)
        se = np.sqrt(edge * (1 - edge) / nTotal.values)
        ax.errorbar(pw.index, edge, yerr=se, marker='o', ms=4, lw=1.2,
                    capsize=2, color='C4', label='edge mutation')
    if 'core' in pw.columns:
        core = pw['core'].reindex(nTotal.index).fillna(0)
        se = np.sqrt(core * (1 - core) / nTotal.values)
        ax.errorbar(pw.index, core, yerr=se, marker='o', ms=4, lw=1.2,
                    capsize=2, color='C3', label='core delivery')

    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel('Fraction of biofilm rescues')
    ax.set_ylim(0, 1)
    ax.set_title('Pathway breakdown of biofilm rescues\n'
                 '(rescueMode = compartment of dominant lineage at rescue)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_sensCore(df, outPath):
    s = summarize(df, ['bCoreRatio', 'dose'])
    fig, ax = plt.subplots(figsize=(7, 5))
    doseGrid = np.linspace(s.dose.min() - 0.01, s.dose.max() + 0.01, 400)
    sGrid = doseGrid - 1.0

    for bc in sorted(s.bCoreRatio.unique()):
        col = BC_COLORS.get(bc, 'k')
        sub = s[s.bCoreRatio == bc].sort_values('dose')
        ax.errorbar(sub.dose, sub.p, yerr=sub.se,
                    marker='o', ms=4, lw=1.2, capsize=2,
                    color=col, label=fr'BF sim, $b_c={bc}$')
        theo = np.array([P_rescue_bf_approx(s_, bc) for s_ in sGrid])
        ax.plot(doseGrid, theo, ls='--', lw=1.2, color=col, alpha=0.8,
                label=fr'BF approx, $b_c={bc}$')

    pwm = np.array([P_rescue_wm_approx(s_) for s_ in sGrid])
    ax.plot(doseGrid, pwm, ls=':', lw=1.4, color='k',
            label='WM approx (reference)')

    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.0)
    ax.set_title('Core dormancy sensitivity (biofilm only)\n'
                 r'($\mu=10^{-4}$, 5000 seeds/point)')
    ax.legend(fontsize=7.5, loc='lower left', ncol=2)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_ratio(df, outPath):
    s = summarize(df, ['condition', 'dose'])
    bf = s[s.condition == 'biofilm']
    wm = s[s.condition == 'wellMixed']
    merged = bf[['dose', 'p', 'se']].merge(
        wm[['dose', 'p', 'se']], on='dose', suffixes=('_bf', '_wm'))
    valid = (merged.p_bf > 0) & (merged.p_wm > 0)
    m = merged[valid].sort_values('dose')

    fig, ax = plt.subplots(figsize=(7, 5))
    doseGrid = np.linspace(m.dose.min() - 0.01, m.dose.max() + 0.01, 400)
    sGrid = doseGrid - 1.0
    pwm = np.array([P_rescue_wm_approx(s_) for s_ in sGrid])
    pbf = np.array([P_rescue_bf_approx(s_, B_C_DEF) for s_ in sGrid])
    ax.semilogy(doseGrid, pbf / pwm, 'k--', lw=1.5,
                label=r'analytical $P_{\rm BF}/P_{\rm WM}$ (full approx)')

    ratio = m.p_bf / m.p_wm
    log_se = np.sqrt((m.se_bf / m.p_bf) ** 2 + (m.se_wm / m.p_wm) ** 2)
    lo = ratio * np.exp(-log_se)
    hi = ratio * np.exp(+log_se)
    ax.errorbar(m.dose, ratio, yerr=[ratio - lo, hi - ratio],
                marker='o', ms=5, capsize=3, ls='',
                color='C5', label='simulation')
    ax.axhline(1.0, color='gray', ls=':', lw=1.0)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P_{\rm BF}/P_{\rm WM}$')
    ax.set_title('Biofilm-to-WM rescue ratio: sim vs theory\n'
                 r'($b_c=0.2$, $\mu=10^{-4}$, 5000 seeds/point)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_sensMu(df, outPath):
    s = summarize(df, ['condition', 'mu', 'dose'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for mu in sorted(s.mu.unique()):
        col = MU_COLORS.get(mu, 'k')
        bf = s[(s.condition == 'biofilm') & (s.mu == mu)].sort_values('dose')
        wm = s[(s.condition == 'wellMixed') & (s.mu == mu)].sort_values('dose')
        ax.errorbar(bf.dose, bf.p, yerr=bf.se, marker='o', ms=3, lw=1,
                    capsize=2, color=col, label=fr'BF $\mu={mu:g}$')
        ax.errorbar(wm.dose, wm.p, yerr=wm.se, marker='s', ms=3, lw=1, ls='--',
                    capsize=2, color=col, alpha=0.7,
                    label=fr'WM $\mu={mu:g}$')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_ylim(1e-4, 1.0)
    ax.set_title('Mutation rate sensitivity: P(rescue)\n'
                 r'(BF solid, WM dashed; 5000 seeds/point)')
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    for mu in sorted(s.mu.unique()):
        col = MU_COLORS.get(mu, 'k')
        bf = s[(s.condition == 'biofilm') & (s.mu == mu)].sort_values('dose')
        wm = s[(s.condition == 'wellMixed') & (s.mu == mu)].sort_values('dose')
        merged = bf[['dose', 'p', 'se']].merge(
            wm[['dose', 'p', 'se']], on='dose', suffixes=('_bf', '_wm'))
        m = merged[(merged.p_bf > 0) & (merged.p_wm > 0)].sort_values('dose')
        ratio = m.p_bf / m.p_wm
        log_se = np.sqrt((m.se_bf / m.p_bf) ** 2 + (m.se_wm / m.p_wm) ** 2)
        lo = ratio * np.exp(-log_se)
        hi = ratio * np.exp(+log_se)
        ax.errorbar(m.dose, ratio, yerr=[ratio - lo, hi - ratio],
                    marker='o', ms=4, capsize=2, lw=1.2,
                    color=col, label=fr'$\mu={mu:g}$')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P_{\rm BF}/P_{\rm WM}$')
    ax.set_yscale('log')
    ax.set_title('Mutation rate sensitivity: rescue ratio')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_agreement(df, outPath):
    s = summarize(df, ['condition', 'dose'])
    cwm = s[s.condition == 'chainWM'].sort_values('dose')
    wm = s[s.condition == 'wellMixed'].sort_values('dose')
    m = cwm[['dose', 'p', 'se']].merge(
        wm[['dose', 'p', 'se']], on='dose', suffixes=('_cwm', '_wm'))
    m['diff'] = m.p_cwm - m.p_wm
    m['diffSE'] = np.sqrt(m.se_cwm ** 2 + m.se_wm ** 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(m.dose, m['diff'], yerr=m.diffSE, marker='o', capsize=3,
                color='black')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P_{\rm chainWM}(\mathrm{rescue}) - P_{\rm wellMixed}(\mathrm{rescue})$')
    ax.set_title('Implementation agreement check\n'
                 '(chainWM with l=2000 vs explicit wellMixed)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resultsDir', default=os.path.join(REPO_ROOT, 'results'))
    p.add_argument('--outDir', default=os.path.join(REPO_ROOT, 'figures', 'panels'))
    args = p.parse_args()

    os.makedirs(args.outDir, exist_ok=True)

    print('Loading main.csv...')
    main_df = pd.read_csv(os.path.join(args.resultsDir, 'main.csv'),
                          low_memory=False)
    panel_rescue_curves(main_df, os.path.join(args.outDir, 'sim_rescue_curves.png'))
    panel_pathway(main_df, os.path.join(args.outDir, 'sim_pathway.png'))
    panel_ratio(main_df, os.path.join(args.outDir, 'sim_ratio.png'))
    panel_agreement(main_df, os.path.join(args.outDir, 'sim_agreement.png'))

    print('Loading sensCore.csv...')
    sc_df = pd.read_csv(os.path.join(args.resultsDir, 'sensCore.csv'),
                        low_memory=False)
    panel_sensCore(sc_df, os.path.join(args.outDir, 'sim_sensCore.png'))

    print('Loading sensMu.csv...')
    sm_df = pd.read_csv(os.path.join(args.resultsDir, 'sensMu.csv'),
                        low_memory=False)
    panel_sensMu(sm_df, os.path.join(args.outDir, 'sim_sensMu.png'))


if __name__ == '__main__':
    main()
