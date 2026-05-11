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
        ax.plot(sub.dose, sub.p,
                    marker='o', ms=4, lw=1.2,
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
        ax.plot(pw.index, edge, marker='o', ms=4, lw=1.2, color='C4', label='edge mutation')
    if 'core' in pw.columns:
        core = pw['core'].reindex(nTotal.index).fillna(0)
        se = np.sqrt(core * (1 - core) / nTotal.values)
        ax.plot(pw.index, core, marker='o', ms=4, lw=1.2, color='C3', label='core delivery')

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
        ax.plot(sub.dose, sub.p,
                    marker='o', ms=4, lw=1.2,
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
    ax.plot(m.dose, ratio,
                marker='o', ms=5, ls='',
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
        ax.plot(bf.dose, bf.p, marker='o', ms=3, lw=1, color=col, label=fr'BF $\mu={mu:g}$')
        ax.plot(wm.dose, wm.p, marker='s', ms=3, lw=1, ls='--', color=col, alpha=0.7,
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
        ax.plot(m.dose, ratio,
                    marker='o', ms=4, lw=1.2,
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
    ax.plot(m.dose, m['diff'], marker='o',
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


def panel_mutation_supply(df, outPath):
    '''narrative.tex Sec 4: biofilm should have ~1.8x more mutations than WM
    (active core adds extra births). Plots mean total mutations per run.'''
    df = df.copy()
    df['nMutTotal'] = df['nMutCore'].fillna(0) + df['nMutEdge'].fillna(0)
    g = (df.groupby(['condition', 'dose'])
           .agg(meanMut=('nMutTotal', 'mean'),
                seMut=('nMutTotal', lambda x: x.std() / np.sqrt(len(x))))
           .reset_index())
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in ['biofilm', 'chainWM', 'wellMixed']:
        s = g[g.condition == cond].sort_values('dose')
        if len(s) == 0:
            continue
        ax.plot(s.dose, s.meanMut,
                    marker='o', ms=4, lw=1.2,
                    color=CONDITION_COLORS[cond], label=cond)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$\langle$ R lineages produced per run $\rangle$')
    ax.set_title('Mutation supply vs dose (narrative Sec 4)\n'
                 'BF should exceed WM at active core (default $b_c=0.2$)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_estab_phase(df, outPath):
    '''narrative.tex Sec 5: BF Phase-1-born edge mutations face WT resupply
    suppression; BF Phase-2-born edge mutations should establish at near-WM
    rate (no resupply). The BF P1 vs BF P2 vs WM comparison is the central
    test of the analytical framework.'''
    rows = []
    for (cond, dose), sub in df.groupby(['condition', 'dose']):
        if cond == 'biofilm':
            for phase, mutCol, estCol in [('P1', 'nMutEdgePhase1', 'nEstEdgePhase1'),
                                           ('P2', 'nMutEdgePhase2', 'nEstEdgePhase2')]:
                nMut = sub[mutCol].sum()
                nEst = sub[estCol].sum()
                if nMut > 0:
                    p = nEst / nMut
                    se = np.sqrt(p * (1 - p) / nMut)
                    rows.append({'group': f'BF {phase}', 'dose': dose,
                                 'p': p, 'se': se, 'n': nMut})
        elif cond == 'wellMixed':
            nMut = sub['nMutEdge'].sum()
            nEst = sub['nEstablishedEdge'].sum()
            if nMut > 0:
                p = nEst / nMut
                se = np.sqrt(p * (1 - p) / nMut)
                rows.append({'group': 'WM', 'dose': dose,
                             'p': p, 'se': se, 'n': nMut})
    g = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {'BF P1': ('C0', 'o', '-'),
              'BF P2': ('C0', 's', '--'),
              'WM':    ('C2', '^', '-')}
    for grp in ['BF P1', 'BF P2', 'WM']:
        sub = g[g.group == grp].sort_values('dose')
        if len(sub) == 0:
            continue
        col, mk, ls = styles[grp]
        ax.plot(sub.dose, sub.p,
                    marker=mk, ms=5, lw=1.4, ls=ls,
                    color=col, label=grp)
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel('Establishment fraction (lineages reaching $k_{est}$)')
    ax.set_title('Establishment by phase (narrative Sec 5)\n'
                 'BF P1 << BF P2 ~ WM is the WT-resupply suppression signal')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_delivery(df, outPath):
    '''narrative.tex Sec 6: drift kills core mutations at low dose; delivery
    fraction (nDelivered / nMutCore) rises with dose for the biofilm.'''
    bf = df[df.condition == 'biofilm']
    rows = []
    for dose, sub in bf.groupby('dose'):
        nMut = sub['nMutCore'].sum()
        nDel = sub['nDelivered'].sum()
        if nMut > 0:
            p = nDel / nMut
            se = np.sqrt(p * (1 - p) / nMut)
            rows.append({'dose': dose, 'p': p, 'se': se, 'n': nMut})
    g = pd.DataFrame(rows).sort_values('dose')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g.dose, g.p,
                marker='o', ms=5, lw=1.4,
                color='C3', label='BF: core lineages reaching edge')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel('Delivery fraction (nDelivered / nMutCore)')
    ax.set_title('Core-to-edge delivery efficiency vs dose (narrative Sec 6)\n'
                 'Drift suppression at low dose; efficient at high dose')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_bolus(df, outPath):
    '''narrative.tex Sec 8: a core lineage that survives drift grows in the
    core before delivery, arriving as a bolus. meanDeliverySizeCore should
    rise with dose (signature of bolus advantage).'''
    bf = df[(df.condition == 'biofilm') & df['meanDeliverySizeCore'].notna()]
    g = (bf.groupby('dose')
           .agg(meanBolus=('meanDeliverySizeCore', 'mean'),
                seBolus=('meanDeliverySizeCore', lambda x: x.std() / np.sqrt(len(x))),
                n=('meanDeliverySizeCore', 'count'))
           .reset_index())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g.dose, g.meanBolus,
                marker='o', ms=5, lw=1.4,
                color='C5', label='mean delivery size (core-born)')
    ax.axhline(1.0, color='gray', ls=':', lw=1, label='single-cell delivery (no bolus)')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$\langle$ lineage size at first edge entry $\rangle$')
    ax.set_title('Bolus size of core-delivered lineages vs dose (narrative Sec 8)\n'
                 'Rising values are the bolus-advantage signature')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_rescue_time(df, outPath):
    '''narrative.tex "What we don't know": time-to-rescue as alternative
    signal. BF should be slower at low dose (edge establishment suppressed)
    and possibly faster at high dose (bolus delivery).'''
    rescued = df[df.rescued == 1].copy()
    rescued['rescueTime'] = pd.to_numeric(rescued['rescueTime'], errors='coerce')
    g = (rescued.dropna(subset=['rescueTime'])
                .groupby(['condition', 'dose'])
                .agg(median=('rescueTime', 'median'),
                     q25=('rescueTime', lambda x: x.quantile(0.25)),
                     q75=('rescueTime', lambda x: x.quantile(0.75)),
                     n=('rescueTime', 'count'))
                .reset_index())

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in ['biofilm', 'chainWM', 'wellMixed']:
        sub = g[g.condition == cond].sort_values('dose')
        if len(sub) == 0:
            continue
        ax.plot(sub.dose, sub['median'],
                    marker='o', ms=4, lw=1.2,
                    color=CONDITION_COLORS[cond], label=f'{cond} (median)')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel('Time to rescue (rescued runs only)')
    ax.set_yscale('log')
    ax.set_title('Time to rescue vs dose (narrative "What we don\'t know")\n'
                 'BF slower at low dose, comparable/faster at high dose')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def _aggRescue(df, groupKeys):
    if not groupKeys:
        n = len(df)
        nR = int(df['rescued'].sum())
        p = nR / n if n else float('nan')
        return pd.DataFrame([{'n': n, 'nRescued': nR, 'p': p}])
    g = df.groupby(groupKeys).agg(n=('seed', 'count'),
                                   nRescued=('rescued', 'sum')).reset_index()
    g['p'] = g.nRescued / g.n
    return g


def _aggEstab(df, groupKeys):
    '''Aggregate establishment fraction: total edge lineages reaching k_est
    divided by total edge mutations, across all runs in the group.'''
    if not groupKeys:
        nEst = int(df['nEstablishedEdge'].sum())
        nMut = int(df['nMutEdge'].sum())
        p = nEst / nMut if nMut else float('nan')
        return pd.DataFrame([{'nEst': nEst, 'nMut': nMut, 'p': p}])
    g = df.groupby(groupKeys).agg(nEst=('nEstablishedEdge', 'sum'),
                                   nMut=('nMutEdge', 'sum')).reset_index()
    g['p'] = np.where(g.nMut > 0, g.nEst / g.nMut.replace(0, np.nan), np.nan)
    return g


def panel_double_edge_rescue(main_df, sensCore_df, outPath):
    '''Biofilm as a double-edged sword (P(rescue)).

    Compares BF at multiple b_c values against the WM reference vs dose.
    The dormant core (b_c=0) is uniformly worse than WM at every dose (no
    crossover). An active core (b_c>0) crosses above WM at some dose; the
    crossover shifts earlier with higher b_c.
    '''
    bf = _aggRescue(sensCore_df, ['dose', 'bCoreRatio'])
    wm = _aggRescue(main_df[main_df.condition == 'wellMixed'], ['dose']).sort_values('dose')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bcVals = sorted(bf.bCoreRatio.unique())
    cmap = plt.get_cmap('viridis')
    for i, bc in enumerate(bcVals):
        sub = bf[bf.bCoreRatio == bc].sort_values('dose')
        col = cmap(i / max(1, len(bcVals) - 1))
        ax.plot(sub.dose, sub.p,
                    marker='o', ms=4, lw=1.2,
                    color=col, label=fr'BF, $b_c={bc}$')

    ax.plot(wm.dose, wm.p,
                marker='s', ms=6, lw=2.5,
                color='red', label='wellMixed (reference)', zorder=5)

    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_title('Biofilm as a double-edged sword: P(rescue) crossover\n'
                 r'Dormant ($b_c=0$) BF is uniformly worse; active BF crosses above WM with dose')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_double_edge_est(main_df, sensCore_df, outPath):
    '''Same double-edged sword view, on per-mutation establishment.

    P(establishment | edge mutation) = (edge lineages reaching k_est) /
    (edge lineages appearing). Decouples mutation supply from the
    selection-dependent establishment bottleneck.
    '''
    bf = _aggEstab(sensCore_df, ['dose', 'bCoreRatio'])
    wm = _aggEstab(main_df[main_df.condition == 'wellMixed'], ['dose']).sort_values('dose')

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bcVals = sorted(bf.bCoreRatio.unique())
    cmap = plt.get_cmap('viridis')
    for i, bc in enumerate(bcVals):
        sub = bf[bf.bCoreRatio == bc].sort_values('dose')
        col = cmap(i / max(1, len(bcVals) - 1))
        ax.plot(sub.dose, sub.p,
                    marker='o', ms=4, lw=1.2,
                    color=col, label=fr'BF, $b_c={bc}$')

    ax.plot(wm.dose, wm.p,
                marker='s', ms=6, lw=2.5,
                color='red', label='wellMixed (reference)', zorder=5)

    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{establishment} \mid \mathrm{edge\ mutation})$')
    ax.set_yscale('log')
    ax.set_title('Per-mutation establishment: BF vs WM across dose and $b_c$\n'
                 'WT resupply in Phase 1 keeps BF below WM at low dose; gap closes at high dose')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_double_edge_bars(main_df, sensCore_df, outPath,
                           lowDose=1.05, highDose=2.8,
                           dormantBc=0.0, activeBc=0.8):
    '''Crisp 2x2 bar summary of the double-edged sword.

    Rows: P(rescue) and P(establishment | edge mutation).
    Cols: 4 corner cells: (low dose, dormant), (low dose, active),
           (high dose, dormant), (high dose, active).
    Each cell shows BF vs WM bars side by side.
    '''
    cells = [
        (lowDose,  dormantBc, f'low dose ({lowDose})\ndormant core ($b_c={dormantBc}$)'),
        (lowDose,  activeBc,  f'low dose ({lowDose})\nactive core ($b_c={activeBc}$)'),
        (highDose, dormantBc, f'high dose ({highDose})\ndormant core ($b_c={dormantBc}$)'),
        (highDose, activeBc,  f'high dose ({highDose})\nactive core ($b_c={activeBc}$)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.35
    xs = np.arange(len(cells))
    labels = [c[2] for c in cells]

    for ax, metric, fn, ylabel, title in [
        (axes[0], 'rescue', _aggRescue, r'$P(\mathrm{rescue})$',
         'P(rescue) at four extremes'),
        (axes[1], 'estab',  _aggEstab,  r'$P(\mathrm{est} \mid \mathrm{edge\ mut})$',
         'P(establishment) at four extremes'),
    ]:
        bf_p, wm_p = [], []
        for dose, bc, _ in cells:
            bf_sub = sensCore_df[(sensCore_df.dose == dose) &
                                 (sensCore_df.bCoreRatio == bc)]
            wm_sub = main_df[(main_df.condition == 'wellMixed') &
                             (main_df.dose == dose)]
            bf_g = fn(bf_sub, []).iloc[0]
            wm_g = fn(wm_sub, []).iloc[0]
            bf_p.append(bf_g['p'])
            wm_p.append(wm_g['p'])
        ax.bar(xs - width/2, bf_p, width,
               color=CONDITION_COLORS['biofilm'], label='biofilm')
        ax.bar(xs + width/2, wm_p, width,
               color=CONDITION_COLORS['wellMixed'], label='wellMixed')
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3, which='both', axis='y')

    fig.suptitle('Biofilm as a double-edged sword: low dose / high dose, dormant / active core',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  wrote {outPath}')


def panel_sensL(df, outPath):
    '''narrative.tex Sec 8: crossover dose depends on edge width l.'''
    s = (df.groupby(['l', 'dose'])
           .agg(n=('seed', 'count'), nRescued=('rescued', 'sum'))
           .reset_index())
    s['p'] = s.nRescued / s.n
    s['se'] = np.sqrt(s.p * (1 - s.p) / s.n)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    lVals = sorted(df.l.unique())
    for i, lVal in enumerate(lVals):
        sub = s[s.l == lVal].sort_values('dose')
        col = cmap(i / max(1, len(lVals) - 1))
        ax.plot(sub.dose, sub.p,
                    marker='o', ms=4, lw=1.2,
                    color=col, label=f'l = {int(lVal)}')
    ax.set_xlabel(r'dose $(d_e)$')
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_title('Edge-width sensitivity (narrative Sec 8)\n'
                 r'Crossover dose should shift with $l$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
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

    mainPath = os.path.join(args.resultsDir, 'main.csv')
    if os.path.exists(mainPath):
        print('Loading main.csv...')
        main_df = pd.read_csv(mainPath, low_memory=False)
        panel_rescue_curves(main_df, os.path.join(args.outDir, 'sim_rescue_curves.png'))
        panel_pathway(main_df, os.path.join(args.outDir, 'sim_pathway.png'))
        panel_ratio(main_df, os.path.join(args.outDir, 'sim_ratio.png'))
        panel_agreement(main_df, os.path.join(args.outDir, 'sim_agreement.png'))
        panel_mutation_supply(main_df, os.path.join(args.outDir, 'sim_mutation_supply.png'))
        panel_delivery(main_df, os.path.join(args.outDir, 'sim_delivery.png'))
        panel_bolus(main_df, os.path.join(args.outDir, 'sim_bolus.png'))
        panel_rescue_time(main_df, os.path.join(args.outDir, 'sim_rescue_time.png'))
        if 'nMutEdgePhase1' in main_df.columns:
            panel_estab_phase(main_df, os.path.join(args.outDir, 'sim_estab_phase.png'))
        else:
            print('  (skipping sim_estab_phase: phase columns not in CSV; rerun sweep)')

    scPath = os.path.join(args.resultsDir, 'sensCore.csv')
    if os.path.exists(scPath):
        print('Loading sensCore.csv...')
        sc_df = pd.read_csv(scPath, low_memory=False)
        panel_sensCore(sc_df, os.path.join(args.outDir, 'sim_sensCore.png'))
        # Double-edged-sword panels combine wellMixed (main) and BF-vs-bcore (sensCore).
        if os.path.exists(mainPath):
            panel_double_edge_rescue(main_df, sc_df,
                os.path.join(args.outDir, 'sim_double_edge_rescue.png'))
            panel_double_edge_est(main_df, sc_df,
                os.path.join(args.outDir, 'sim_double_edge_est.png'))
            panel_double_edge_bars(main_df, sc_df,
                os.path.join(args.outDir, 'sim_double_edge_bars.png'))

    smPath = os.path.join(args.resultsDir, 'sensMu.csv')
    if os.path.exists(smPath):
        print('Loading sensMu.csv...')
        sm_df = pd.read_csv(smPath, low_memory=False)
        panel_sensMu(sm_df, os.path.join(args.outDir, 'sim_sensMu.png'))

    slPath = os.path.join(args.resultsDir, 'sensL.csv')
    if os.path.exists(slPath):
        print('Loading sensL.csv...')
        sl_df = pd.read_csv(slPath, low_memory=False)
        panel_sensL(sl_df, os.path.join(args.outDir, 'sim_sensL.png'))


if __name__ == '__main__':
    main()
