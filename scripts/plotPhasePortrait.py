'''Phase-portrait heatmaps from runSweepLocal phase sweeps.

For each phase sweep, the headline figure is a heatmap of
log10(P_BF / P_WM) over the 2D parameter plane, with diverging colormap
centered at zero (BF = WM). Below the heatmap, a panel of marginal P(rescue)
curves and a panel of secondary observables (W extinction probability,
mean nLineagesEstablished, mean nLineagesAtRescue) gives context.

No error bars (these are Monte Carlo estimators of the rescue probability,
not measurements with uncertainty in the data-collection sense).
No theoretical overlay (the writeup analytics do not predict the simulation
result and overlaying them is misleading).

Usage:
    python3 scripts/plotPhasePortrait.py \\
        --csv results/phaseBc.csv --x dose --y bCoreRatio --out figures/phaseBc.png
    python3 scripts/plotPhasePortrait.py \\
        --csv results/phaseL.csv  --x dose --y l         --out figures/phaseL.png
    python3 scripts/plotPhasePortrait.py \\
        --csv results/phaseMu.csv --x dose --y mu        --out figures/phaseMu.png
'''
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def summarize(df, groupKeys):
    g = (df.groupby(groupKeys, dropna=False)
           .agg(n=('seed', 'count'),
                pRescue=('rescued', 'mean'),
                pWtExt=('wtExtinct', 'mean'),
                meanLinEst=('nLineagesEstablished', 'mean'),
                meanLinAtRescue=('nLineagesAtRescue', 'mean'),
                meanFinalN=('finalN', 'mean'))
           .reset_index())
    return g


def panel_ratio_heatmap(ax, df, xCol, yCol):
    bf = df[df.condition == 'biofilm']
    wm = df[df.condition == 'wellMixed']
    bfSummary = summarize(bf, [xCol, yCol])
    wmSummary = summarize(wm, [xCol])  # WM has only one curve in dose

    bfPivot = bfSummary.pivot(index=yCol, columns=xCol, values='pRescue')
    bfPivot = bfPivot.sort_index(ascending=False)

    # Broadcast WM across yCol axis
    wmByDose = wmSummary.set_index(xCol)['pRescue']
    ratio = bfPivot.div(wmByDose, axis=1)
    logRatio = np.log10(ratio.replace(0, np.nan))

    vmax = max(0.1, np.nanmax(np.abs(logRatio.values)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(logRatio.values, aspect='auto', cmap='RdBu_r', norm=norm,
                   interpolation='nearest')
    ax.set_xticks(range(len(bfPivot.columns)))
    ax.set_xticklabels([f'{d:.3g}' for d in bfPivot.columns], rotation=45,
                       fontsize=8)
    ax.set_yticks(range(len(bfPivot.index)))
    ax.set_yticklabels([f'{y:.3g}' for y in bfPivot.index], fontsize=8)
    ax.set_xlabel(xCol)
    ax.set_ylabel(yCol)
    ax.set_title(r'$\log_{10}(P_{\rm BF}/P_{\rm WM})$  (red: BF wins, blue: WM wins)')

    # Overlay numeric values
    for i, yLabel in enumerate(bfPivot.index):
        for j, xLabel in enumerate(bfPivot.columns):
            v = logRatio.iloc[i, j]
            if pd.notnull(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(v) < 0.5 * vmax else 'white')
    plt.colorbar(im, ax=ax, label=r'$\log_{10}(P_{\rm BF}/P_{\rm WM})$')


def panel_pRescue_marginal(ax, df, xCol, yCol):
    bf = df[df.condition == 'biofilm']
    wm = df[df.condition == 'wellMixed']
    bfSummary = summarize(bf, [xCol, yCol]).sort_values([yCol, xCol])
    wmSummary = summarize(wm, [xCol]).sort_values(xCol)

    cmap = plt.cm.viridis
    yVals = sorted(bfSummary[yCol].dropna().unique())
    for i, y in enumerate(yVals):
        sub = bfSummary[bfSummary[yCol] == y]
        col = cmap(i / max(1, len(yVals) - 1))
        ax.plot(sub[xCol], sub.pRescue, marker='o', ms=4, lw=1.2,
                color=col, label=f'BF {yCol}={y:.3g}')
    ax.plot(wmSummary[xCol], wmSummary.pRescue, marker='s', ms=5, lw=2,
            color='black', label='WM')
    ax.set_xlabel(xCol)
    ax.set_ylabel(r'$P(\mathrm{rescue})$')
    ax.set_yscale('log')
    ax.set_xscale('linear')
    ax.set_title('Rescue probability (no CIs; Monte Carlo estimator)')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(alpha=0.3, which='both')


def panel_observables(ax, df, xCol, yCol):
    '''Right panel: secondary observables for biofilm at one representative yCol slice
    (the largest value), and WM, vs xCol.'''
    bf = df[df.condition == 'biofilm']
    wm = df[df.condition == 'wellMixed']
    bfSummary = summarize(bf, [xCol, yCol])
    wmSummary = summarize(wm, [xCol])

    yVals = sorted(bfSummary[yCol].dropna().unique())
    yPick = yVals[-1]  # largest slice as representative
    bfRow = bfSummary[bfSummary[yCol] == yPick].sort_values(xCol)

    ax2 = ax.twinx()
    ax.plot(bfRow[xCol], bfRow.pWtExt, 'o-', ms=4, color='C2',
            label=f'BF {yCol}={yPick:.3g}: P(WT extinct)')
    ax.plot(wmSummary[xCol], wmSummary.pWtExt, 's-', ms=4, color='black',
            label='WM: P(WT extinct)')
    ax.set_xlabel(xCol)
    ax.set_ylabel('P(WT extinct)')
    ax.set_ylim(0, 1.05)

    ax2.plot(bfRow[xCol], bfRow.meanLinEst, '^--', ms=4, color='C3', alpha=0.8,
             label=f'BF {yCol}={yPick:.3g}: mean nLineagesEst')
    ax2.plot(wmSummary[xCol], wmSummary.meanLinEst, 'v--', ms=4, color='gray', alpha=0.8,
             label='WM: mean nLineagesEst')
    ax2.set_ylabel('mean nLineagesEstablished', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')
    ax.set_title(f'Secondary observables  (BF slice: {yCol} = {yPick:.3g})')
    ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--x', required=True, help='column for x-axis (typically dose)')
    p.add_argument('--y', required=True, help='column for y-axis (e.g. bCoreRatio, l, mu)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    print(f'Loaded {len(df):,} rows from {args.csv}')
    print(df.groupby('condition').size().to_string())

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    panel_ratio_heatmap(ax0, df, args.x, args.y)
    panel_pRescue_marginal(ax1, df, args.x, args.y)
    panel_observables(ax2, df, args.x, args.y)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
