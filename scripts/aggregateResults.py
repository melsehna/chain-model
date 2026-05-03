'''Aggregate sweep results and produce summary figures.

Usage:
    python aggregate.py --resultsDir results --prefix main     --out main.png
    python aggregate.py --resultsDir results --prefix sensCore --out sensCore.png
    python aggregate.py --resultsDir results --prefix sensMu   --out sensMu.png

The main sweep analysis includes a diagnostic panel showing agreement between
chainWM (chain model with l>=N) and wellMixed (explicit well-mixed). These
should be statistically indistinguishable; visible disagreement is a warning.
'''
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONDITION_COLORS = {
    'biofilm':   'C0',
    'chainWM':   'C1',
    'wellMixed': 'C2',
}


def loadResults(resultsDir, prefix):
    pattern = os.path.join(resultsDir, f'{prefix}_*.csv')
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f'No CSVs matching {pattern}')
    print(f'Loading {len(paths)} files...')
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f'Total rows: {len(df)}')
    return df


def summarize(df, groupBy):
    g = df.groupby(groupBy)
    s = g.agg(
        nRuns=('seed', 'count'),
        nRescued=('rescued', 'sum'),
        nExtinct=('extinct', 'sum'),
        meanRescueT=('rescueTime', 'mean'),
        medianRescueT=('rescueTime', 'median'),
    ).reset_index()
    s['pRescue'] = s.nRescued / s.nRuns
    s['pRescueSE'] = np.sqrt(s.pRescue * (1 - s.pRescue) / s.nRuns)
    return s


def plotMain(df, outPath):
    summary = summarize(df, ['condition', 'dose'])
    biofilmRescued = df[(df.condition == 'biofilm') & (df.rescued == 1)]
    pathway = (biofilmRescued.groupby('dose').rescueMode
               .value_counts(normalize=True)
               .unstack(fill_value=0).sort_index().reset_index())

    # 3 panels: all conditions, biofilm pathway, chainWM vs wellMixed agreement
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    for cond in ['biofilm', 'chainWM', 'wellMixed']:
        s = summary[summary.condition == cond].sort_values('dose')
        if len(s) == 0:
            continue
        ax.errorbar(s.dose, s.pRescue, yerr=s.pRescueSE,
                    marker='o', label=cond, capsize=3,
                    color=CONDITION_COLORS[cond])
    ax.set_xlabel('d_WtEdge (dose)')
    ax.set_ylabel('P(rescue)')
    ax.set_title('Rescue probability vs dose')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if 'coreDelivery' in pathway.columns:
        ax.plot(pathway.dose, pathway.coreDelivery, 'o-',
                label='coreDelivery', color='C3')
    if 'edgeMutation' in pathway.columns:
        ax.plot(pathway.dose, pathway.edgeMutation, 'o-',
                label='edgeMutation', color='C4')
    ax.set_xlabel('d_WtEdge (dose)')
    ax.set_ylabel('Fraction of biofilm rescues')
    ax.set_title('Biofilm rescue pathway breakdown')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    # Agreement check: chainWM vs wellMixed
    ax = axes[2]
    cwm = summary[summary.condition == 'chainWM'].sort_values('dose')
    wm = summary[summary.condition == 'wellMixed'].sort_values('dose')
    if len(cwm) > 0 and len(wm) > 0:
        merged = cwm[['dose', 'pRescue', 'pRescueSE']].merge(
            wm[['dose', 'pRescue', 'pRescueSE']],
            on='dose', suffixes=('_chainWM', '_wellMixed'))
        merged['diff'] = merged.pRescue_chainWM - merged.pRescue_wellMixed
        merged['diffSE'] = np.sqrt(merged.pRescueSE_chainWM**2
                                    + merged.pRescueSE_wellMixed**2)
        ax.errorbar(merged.dose, merged['diff'], yerr=merged.diffSE,
                    marker='o', capsize=3, color='black')
        ax.axhline(0, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('d_WtEdge (dose)')
        ax.set_ylabel('P(rescue)_chainWM - P(rescue)_wellMixed')
        ax.set_title('Agreement check\n(chainWM - wellMixed; should be ~0)')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outPath, dpi=120, bbox_inches='tight')
    print(f'Saved {outPath}')


def plotSensCore(df, outPath):
    summary = summarize(df, ['dose', 'bCoreRatio'])
    fig, ax = plt.subplots(figsize=(8, 5))
    for bCore in sorted(df.bCoreRatio.unique()):
        s = summary[summary.bCoreRatio == bCore].sort_values('dose')
        ax.errorbar(s.dose, s.pRescue, yerr=s.pRescueSE,
                    marker='o', label=f'bCore/bEdge = {bCore}', capsize=3)
    ax.set_xlabel('d_WtEdge')
    ax.set_ylabel('P(rescue)')
    ax.set_title('Biofilm rescue vs core dormancy')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outPath, dpi=120, bbox_inches='tight')
    print(f'Saved {outPath}')


def plotSensMu(df, outPath):
    summary = summarize(df, ['condition', 'dose', 'mu'])
    mus = sorted(df.mu.unique())
    fig, axes = plt.subplots(1, len(mus), figsize=(5 * len(mus), 5), sharey=True)
    if not hasattr(axes, '__len__'):
        axes = [axes]
    for ax, mu in zip(axes, mus):
        sub = summary[summary.mu == mu]
        for cond in ['biofilm', 'chainWM', 'wellMixed']:
            s = sub[sub.condition == cond].sort_values('dose')
            if len(s) == 0:
                continue
            ax.errorbar(s.dose, s.pRescue, yerr=s.pRescueSE,
                        marker='o', label=cond, capsize=3,
                        color=CONDITION_COLORS[cond])
        ax.set_xlabel('d_WtEdge')
        ax.set_title(f'mu = {mu}')
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel('P(rescue)')
    plt.tight_layout()
    plt.savefig(outPath, dpi=120, bbox_inches='tight')
    print(f'Saved {outPath}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resultsDir', default='results')
    p.add_argument('--prefix', required=True,
                   choices=['main', 'sensCore', 'sensMu'])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = loadResults(args.resultsDir, args.prefix)
    df.to_csv(f'{args.prefix}_combined.csv', index=False)
    print(f'Combined data: {args.prefix}_combined.csv')

    if args.prefix == 'main':
        plotMain(df, args.out)
    elif args.prefix == 'sensCore':
        plotSensCore(df, args.out)
    elif args.prefix == 'sensMu':
        plotSensMu(df, args.out)


if __name__ == '__main__':
    main()