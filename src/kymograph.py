'''Kymograph diagnostic plots for the chain model.

The kymograph shows the biofilm's cellular state over time. The x-axis is array
index (0 = innermost, high = outermost); the y-axis is time. Rows are plotted
at their actual (non-uniform) simulation times using pcolormesh, so the step-
plotted boundary lines align exactly with the pixel transitions between colors.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


def stepXY(xVals, tVals):
    '''Convert (x, t) samples into a step-plot polyline matching the pixel
    staircase of an imshow/pcolormesh grid.'''
    xs, ts = [], []
    for i in range(len(xVals) - 1):
        xs.extend([xVals[i], xVals[i]])
        ts.extend([tVals[i], tVals[i + 1]])
    xs.append(xVals[-1])
    ts.append(tVals[-1])
    return np.array(xs), np.array(ts)


def plotKymograph(result, params, title=None, ax=None, figsize=(10, 7)):
    '''Plot a kymograph of the chainModel trajectory.

    Lineage coloring:
      * If the run rescued, every lineage that was in the edge at the rescue
        moment (i.e. every lineage in result['rescueEdgeCounts']) gets a
        distinct saturated color from tab20. All other R lineages -- those
        that appeared and died before rescue, or that never reached the edge,
        or that arrived after -- get a greyscale color so the reader can still
        see their trajectories but they don't compete for visual attention.
      * If the run did not rescue, every lineage gets a tab20 color (no
        greyscale fallback, since there's no rescue set to privilege).
    '''
    traj = result['trajectoryCells']
    times = np.array(result['trajectoryTimes'])
    boundaries = np.array(result['trajectoryBoundaries'])
    outer = np.array([len(s) for s in traj])
    maxN = int(outer.max())
    nSnap = len(traj)

    uniqueRs = sorted({int(x) for s in traj for x in s if x >= 2})

    # Rescuing lineages: those in the edge at rescue moment
    rescueEdgeCounts = result.get('rescueEdgeCounts') or {}
    rescuingLineages = sorted(rescueEdgeCounts.keys())
    rescuingSet = set(rescuingLineages)

    tab20 = plt.get_cmap('tab20')
    paletteSize = tab20.N  

    # Colors for rescuing lineages (saturated tab20) vs others (greyscale)
    paletteOrder = [0, 4, 8, 12, 16,
                    1, 5, 9, 13, 17,
                    2, 6, 10, 14, 18,
                    3, 7, 11, 15, 19]

    lineageColor = {}
    if rescuingSet:
        for i, lin in enumerate(rescuingLineages):
            lineageColor[lin] = tab20(paletteOrder[i % paletteSize])
        nonRescuers = [lin for lin in uniqueRs if lin not in rescuingSet]
        if nonRescuers:
            greys = np.linspace(0.35, 0.75, len(nonRescuers))
            for lin, g in zip(nonRescuers, greys):
                lineageColor[lin] = (g, g, g, 1.0)
    else:
        # No rescue: give every lineage a tab20 color (same striding rule).
        for i, lin in enumerate(uniqueRs):
            lineageColor[lin] = tab20(paletteOrder[i % paletteSize])

    lineageRemap = {lin: i + 3 for i, lin in enumerate(uniqueRs)}
    colors = ['white', '#d0d0d0', '#ffe4b5']
    for lin in uniqueRs:
        colors.append(lineageColor[lin])

    arr = np.zeros((nSnap, maxN), dtype=int)
    for i, s in enumerate(traj):
        bi = boundaries[i]
        for j, g in enumerate(s):
            if g == 1:
                arr[i, j] = 1 if j < bi else 2
            elif g >= 2:
                arr[i, j] = lineageRemap[g]

    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    xEdges = np.arange(maxN + 1)
    yEdges = np.zeros(nSnap + 1)
    yEdges[:nSnap] = times
    if nSnap >= 2:
        yEdges[nSnap] = times[-1] + (times[-1] - times[-2])
    else:
        yEdges[nSnap] = times[-1] + 1.0

    ax.pcolormesh(xEdges, yEdges, arr, cmap=cmap, norm=norm, shading='flat')

    xB, tB = stepXY(boundaries, times)
    xO, tO = stepXY(outer, times)
    ax.plot(xB, tB, color='black', lw=1.2, label='Core/Edge Boundary')
    ax.plot(xO, tO, color='black', lw=1.2, ls='--', label='Outer Biofilm Edge')
    ax.invert_yaxis()

    if result['rescued']:
        ax.axhline(result['rescueTime'], color='red', lw=1.0, ls='--', alpha=0.7)
    if result['extinct']:
        ax.axhline(result['extinctionTime'], color='purple', lw=1.0, ls=':', alpha=0.6)

    ax.set_xlabel('Array Index (0 = Innermost)')
    ax.set_ylabel('Time')
    if title:
        ax.set_title(title)
    ax.set_xlim(0, maxN)

    legendElems = [
        Patch(facecolor='#d0d0d0', edgecolor='k', label='WT (Core)'),
        Patch(facecolor='#ffe4b5', edgecolor='k', label='WT (Edge)'),
        Patch(facecolor='w', edgecolor='k', label='No Biofilm'),
    ]

    if rescuingSet:
        # Rescued: list each rescuing lineage (with its edge count at rescue) in color, then one entry lumping the non-rescuers into grey.
        primary = result.get('primaryLineage')
        for lin in rescuingLineages:
            cnt = rescueEdgeCounts[lin]
            marker = ' [primary]' if lin == primary else ''
            legendElems.append(Patch(
                facecolor=lineageColor[lin], edgecolor='k',
                label=f'R Lineage {lin} (Edge={cnt}){marker}'))
        nonRescuers = [lin for lin in uniqueRs if lin not in rescuingSet]
        if nonRescuers:
            # Use mid-grey swatch for "other R" legend entry
            legendElems.append(Patch(
                facecolor=(0.55, 0.55, 0.55, 1.0), edgecolor='k',
                label=f'Other R Lineages (n={len(nonRescuers)})'))
    else:
        # No rescue: list up to 10 lineages by id, then "... and N more".
        maxInLegend = 10
        for i, lin in enumerate(uniqueRs[:maxInLegend]):
            wrapped = i >= paletteSize
            if wrapped:
                twin = uniqueRs[i % paletteSize]
                label = f'R Lin {lin} (shares color w/ lin {twin})'
            else:
                label = f'R Lineage {lin}'
            legendElems.append(Patch(facecolor=lineageColor[lin], edgecolor='k', label=label))
        if len(uniqueRs) > maxInLegend:
            legendElems.append(Patch(facecolor='white', edgecolor='white',
                                     label=f'... and {len(uniqueRs) - maxInLegend} more'))

    ax.legend(handles=legendElems, loc='lower left', fontsize=8)

    return fig, ax