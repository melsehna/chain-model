'''
Spatial chain model for biofilm antibiotic rescue.

The biofilm is a 1D array of cells. Index 0 is the deep interior; the rightmost
l cells are the edge (antibiotic-exposed). Everything inside is the core. A cell's
birth and death rates depend on which compartment it occupies.

Births insert a daughter next to the parent and shift the array outward; deaths
pop a cell and shift the array inward. The boundary b = max(0, N - l) moves with
N, so an edge death exposes one core cell to the edge and an edge birth pushes one
edge cell into the core. This conveyor-belt flow is what couples the two compartments.

Mutations are replication-linked: with probability mu, a birth produces a new R
lineage daughter instead of inheriting the parent's genotype. R is irreversible.

Initial condition: all WT, biofilm at pre-treatment steady state. At t=0 the edge
WT death rate jumps to its treatment value. Rescue = edge R count reaches rThreshold.
'''

import numpy as np


def sampleEvent(rng, ratesList):
    total = 0.0
    for _, r in ratesList:
        total += r
    if total <= 0.0:
        return None, np.inf
    dt = rng.exponential(1.0 / total)
    threshold = rng.random() * total
    acc = 0.0
    for name, r in ratesList:
        acc += r
        if threshold <= acc:
            return name, dt
    return ratesList[-1][0], dt


def pickPosition(cells, rng, start, end, targetIsWt, wtCount, rCount):
    '''Return the index of a uniformly random cell of the target genotype in
    cells[start:end]. Uses rejection sampling when the target dominates the
    compartment, linear scan otherwise.'''
    targetCount = wtCount if targetIsWt else rCount
    total = wtCount + rCount
    if targetCount == 0:
        raise RuntimeError('pickPosition called with targetCount=0')

    if 4 * targetCount >= total:
        while True:
            i = int(rng.integers(start, end))
            if (cells[i] == 1) == targetIsWt:
                return i

    k = int(rng.integers(targetCount))
    for i in range(start, end):
        if (cells[i] == 1) == targetIsWt:
            if k == 0:
                return i
            k -= 1
    raise RuntimeError('pickPosition: count mismatch')


def simulateChain(params, seed=None, maxGenerations=np.inf, maxTime=np.inf,
                  nMax=None, recordEvery=50, stopAtRescue=False,
                  rThreshold=10, kEst=3, debug=False):
    '''Run the spatial chain model.

    Parameters (dict):
        mu                   mutation rate per birth
        nInit                initial biofilm size
        l                    edge width (antibiotic penetration depth, cells)
        bWtCore, dWtCore     WT birth/death rates in core (equal for steady state)
        bWtEdge              WT birth rate in edge (antibiotic does not affect this)
        dWtEdge              WT death rate in edge (the sweep variable)
        bREdge, dREdge       R birth/death rates in edge
        bRCore, dRCore       R rates in core (default = WT core rates; neutral)
        K                    carrying capacity for the density factor on R births
        densityForm          'linear' (default, Wilson), 'step' (K as a ceiling
                             only), or 'none' (no cap)

    Rescue: rEdge >= rThreshold (default 10). Defaults stop only on rescue
    (if stopAtRescue) or extinction (N=0); maxGenerations and maxTime default
    to np.inf since the model is guaranteed to terminate naturally. nMax
    (default 10*nInit) remains a bug-detector for runaway-N situations.

    If debug is True, compartment counts are checked against the cells array
    after every event.

    Lineage tracking: every R lineage that ever appears is recorded in the
    internal `lineages` dict. Aggregate counts over this dict are returned.
    '''
    rng = np.random.default_rng(seed)

    mu    = float(params['mu'])
    nInit = int(params['nInit'])
    l     = params.get('l')
    l     = None if l is None else int(l)
    bWtC  = float(params['bWtCore'])
    dWtC  = float(params['dWtCore'])
    bRC   = float(params.get('bRCore', bWtC))
    dRC   = float(params.get('dRCore', dWtC))
    bWtE  = float(params['bWtEdge'])
    dWtE  = float(params['dWtEdge'])
    bRE   = float(params['bREdge'])
    dRE   = float(params['dREdge'])
    K     = float(params.get('K', nInit))  # Wilson carrying capacity for R births

    # Penetration-limited dose profile (optional). A fixed l assumes the exposed
    # layer has the same thickness whatever the applied dose, which cannot happen:
    # raising the drug pushes the kill front deeper. With attenuation
    # c(z) = c0 * exp(-z / lambdaPen) and the MIC at dWtEdge == bWtEdge (net edge
    # growth zero), cells are above MIC down to z* = lambdaPen * ln(dose / MIC), so
    #     l = lambdaPen * ln(dWtEdge / bWtEdge),
    # capped at nInit (the drug reaches everything; no core forms). Omit lambdaPen
    # to use the fixed l passed in params, which is the default and leaves every
    # existing run unchanged.
    lambdaPen = params.get('lambdaPen')
    if lambdaPen is not None:
        if dWtE <= bWtE:
            raise ValueError('lambdaPen requires a super-MIC dose (dWtEdge > bWtEdge); '
                             f'got dWtEdge={dWtE}, bWtEdge={bWtE}')
        l = int(round(float(lambdaPen) * np.log(dWtE / bWtE)))
        l = max(1, min(l, nInit))
    if l is None:
        raise ValueError("params must contain 'l', or 'lambdaPen' to derive it from dose")
    densityForm = params.get('densityForm', 'linear')
    if densityForm not in ('linear', 'step', 'none'):
        raise ValueError(f"densityForm must be 'linear', 'step' or 'none', got {densityForm!r}")

    if nMax is None:
        nMax = 10 * nInit

    cells = [1] * nInit
    N = nInit
    b = max(0, N - l)

    wtCore = b
    wtEdge = N - b
    rCore  = 0
    rEdge  = 0

    time = 0.0
    generations = 0
    recordCounter = 0

    # Phase 1 = N > l, Phase 2 = N <= l 
    phase1EndTime = 0.0 if nInit <= l else None

    rescued = False
    rescueTime = None
    rescueGeneration = None
    rescueEdgeCounts = None  # populated at rescue-trigger moment; see below

    wtExtinct = False
    wtExtinctTime = None

    nextLineage = 2
    mutationEvents = []

    # Per-lineage bookkeeping. Entries are never removed; when a lineage goes extinct we just record deathTime and stop updating its count. liveCount: current count in population (0 if extinct) birthRegion: 'core' or 'edge' (compartment at mutation) everReachedEdge: True iff the lineage has ever had >= 1 cell in the edge
    
    lineages = {}

    # Mirror of `cells` for R genotypes only: dict mapping lineage_id -> liveCount. Maintained incrementally; lineages[id]['liveCount'] reads from here. Use aux dict so we can drop zero keys without losing the metadata stored in `lineages` (birthRegion, birthTime, etc).

    def recordMutation(lineageId, bornInCore, atTime, atGeneration, position):
        region = 'core' if bornInCore else 'edge'
        mutationEvents.append({
            'lineage': lineageId,
            'time': atTime,
            'generation': atGeneration,
            'region': region,
            'position': position,
        })
        lineages[lineageId] = {
            'birthRegion': region,
            'birthTime': atTime,
            'birthGeneration': atGeneration,
            'liveCount': 1,
            'maxLiveCount': 1,
            'everReachedEdge': not bornInCore,  # edge-born starts in edge
            'everEnteredCore': bornInCore,      # core-born starts in core
            'nSweptToCore': 0,                  # edge -> core reclassifications of this lineage's cells
            'deliverySize': 1 if not bornInCore else None,  # set on first edge entry for core-born
            'deathTime': None,
            'deathGeneration': None,
        }

    def incLineage(lineageId):
        info = lineages[lineageId]
        info['liveCount'] += 1
        if info['liveCount'] > info['maxLiveCount']:
            info['maxLiveCount'] = info['liveCount']

    def decLineage(lineageId, atTime, atGeneration):
        info = lineages[lineageId]
        info['liveCount'] -= 1
        if info['liveCount'] == 0:
            info['deathTime'] = atTime
            info['deathGeneration'] = atGeneration

    trajectoryCells = [list(cells)]
    trajectoryTimes = [time]
    trajectoryBoundaries = [b]
    trajectoryGenerations = [generations]

    terminationReason = None

    def checkInvariants(where):
        bLocal = max(0, len(cells) - l)
        actWC = sum(1 for c in cells[:bLocal] if c == 1)
        actRC = sum(1 for c in cells[:bLocal] if c >= 2)
        actWE = sum(1 for c in cells[bLocal:] if c == 1)
        actRE = sum(1 for c in cells[bLocal:] if c >= 2)
        assert actWC == wtCore, f'{where}: wtCore {wtCore} != actual {actWC}'
        assert actRC == rCore,  f'{where}: rCore {rCore} != actual {actRC}'
        assert actWE == wtEdge, f'{where}: wtEdge {wtEdge} != actual {actWE}'
        assert actRE == rEdge,  f'{where}: rEdge {rEdge} != actual {actRE}'
        assert len(cells) == N, f'{where}: len(cells) {len(cells)} != N {N}'
        # Cross-check per-lineage counts against cells
        fromCells = {}
        for c in cells:
            if c >= 2:
                fromCells[c] = fromCells.get(c, 0) + 1
        for lin, info in lineages.items():
            expected = fromCells.get(lin, 0)
            assert info['liveCount'] == expected, (
                f'{where}: lineage {lin} liveCount {info["liveCount"]} != actual {expected}'
            )
        assert sum(info['liveCount'] for info in lineages.values()) == rCore + rEdge, (
            f'{where}: lineage total != rCore+rEdge'
        )

    while True:
        if generations >= maxGenerations:
            terminationReason = 'maxGenerations'
            break
        if time >= maxTime:
            terminationReason = 'maxTime'
            break
        if N <= 0:
            terminationReason = 'extinction'
            break
        if N > nMax:
            terminationReason = 'nMaxExceeded'
            break

        b = max(0, N - l)

        # Density factor on R births only (WT births stay density-independent per Wilson 2017 / Uecker 2014 D=1 model). Applied to R in both compartments using the global (wt+r)/K.
        # 'linear' is Wilson's logistic factor: it scales R growth across the whole density range, so with K = nInit a core R cell (bRC = dRC nominally) is subcritical at every N, and no R grows anywhere until N < K*(1 - dR/bR). Equilibrium after rescue is K*(1 - dR/bR).
        # 'step' treats K as a ceiling only: no effect below it, births blocked at it. Core R stays exactly critical and edge R keeps sR = bR - dR throughout the decline; the rescued population saturates at K.
        # 'none' removes the cap entirely; only safe when runs stop at rescue.
        if densityForm == 'linear':
            densityFactor = max(0.0, 1.0 - N / K)
        elif densityForm == 'step':
            densityFactor = 1.0 if N < K else 0.0
        else:
            densityFactor = 1.0

        ratesList = (
            ('birthWtCore', bWtC * wtCore),
            ('birthRCore',  bRC  * rCore * densityFactor),
            ('birthWtEdge', bWtE * wtEdge),
            ('birthREdge',  bRE  * rEdge * densityFactor),
            ('deathWtCore', dWtC * wtCore),
            ('deathRCore',  dRC  * rCore),
            ('deathWtEdge', dWtE * wtEdge),
            ('deathREdge',  dRE  * rEdge),
        )

        event, dt = sampleEvent(rng, ratesList)
        if event is None:
            terminationReason = 'zeroRate'
            break

        time += dt
        generations += 1

        isBirth = event[0] == 'b'
        isWt = 'Wt' in event
        inCore = 'Core' in event

        if isBirth:
            if inCore:
                cStart, cEnd = 0, b
                wtN, rN = wtCore, rCore
            else:
                cStart, cEnd = b, N
                wtN, rN = wtEdge, rEdge
            idx = pickPosition(cells, rng, cStart, cEnd, isWt, wtN, rN)

            if isWt and rng.random() < mu:
                daughterGenotype = nextLineage
                recordMutation(nextLineage, inCore, time, generations, idx)
                nextLineage += 1
            else:
                daughterGenotype = cells[idx]
                if daughterGenotype >= 2:
                    incLineage(daughterGenotype)

            offset = int(rng.integers(0, 2))
            p = idx + offset
            cells.insert(p, daughterGenotype)
            N += 1
            newB = max(0, N - l)

            if p < newB:
                if daughterGenotype == 1: wtCore += 1
                else:
                    rCore += 1
                    lineages[daughterGenotype]['everEnteredCore'] = True
            else:
                if daughterGenotype == 1: wtEdge += 1
                else:                     rEdge += 1
                # Daughter was placed in edge -- if it's R, it has now "reached edge"
                if daughterGenotype >= 2:
                    info = lineages[daughterGenotype]
                    if not info['everReachedEdge']:
                        info['deliverySize'] = info['liveCount']
                    info['everReachedEdge'] = True

            if newB > b and p > b:
                # The insert at p > b did not displace cells[b], which now sits just inside the new boundary and reclassifies edge -> core.
                bt = cells[b]
                if bt == 1:
                    wtEdge -= 1
                    wtCore += 1
                else:
                    rEdge -= 1
                    rCore += 1
                    # bt is now in core, but this is a compartment move, not a loss from edge-exposure history. No everReachedEdge change.
                    # It does matter when R is not neutral in the core (bRCore < dRCore): the conveyor is pushing the lineage into a compartment where it decays.
                    infoSwept = lineages[bt]
                    infoSwept['everEnteredCore'] = True
                    infoSwept['nSweptToCore'] += 1

        else:
            if inCore:
                cStart, cEnd = 0, b
                wtN, rN = wtCore, rCore
            else:
                cStart, cEnd = b, N
                wtN, rN = wtEdge, rEdge
            q = pickPosition(cells, rng, cStart, cEnd, isWt, wtN, rN)

            newB = max(0, (N - 1) - l)
            doTransition = (newB < b) and (q >= b)
            btDeath = cells[b - 1] if doTransition else None
            deadGenotype = cells[q]

            cells.pop(q)
            N -= 1

            if inCore:
                if isWt: wtCore -= 1
                else:    rCore  -= 1
            else:
                if isWt: wtEdge -= 1
                else:    rEdge  -= 1

            if isWt and (wtCore + wtEdge) == 0 and not wtExtinct:
                wtExtinct = True
                wtExtinctTime = time

            if deadGenotype >= 2:
                decLineage(deadGenotype, time, generations)

            if doTransition:
                # An edge death at q >= b pops without displacing cells[b-1], which now sits in the edge and reclassifies core -> edge.
                if btDeath == 1:
                    wtCore -= 1
                    wtEdge += 1
                else:
                    rCore -= 1
                    rEdge += 1
                    # btDeath (an R lineage) has now reached the edge
                    info = lineages[btDeath]
                    if not info['everReachedEdge']:
                        info['deliverySize'] = info['liveCount']
                    info['everReachedEdge'] = True

        if debug:
            checkInvariants(f'after {event}')

        if phase1EndTime is None and N <= l:
            phase1EndTime = time

        if (not rescued) and rEdge >= rThreshold:
            rescued = True
            rescueTime = time
            rescueGeneration = generations

            # Capture edge lineage counts at the rescue moment
            bAtRescue = max(0, N - l)
            rescueEdgeCounts = {}
            for c in cells[bAtRescue:]:
                if c >= 2:
                    rescueEdgeCounts[c] = rescueEdgeCounts.get(c, 0) + 1

            if stopAtRescue:
                terminationReason = 'rescue'
                break

        recordCounter += 1
        if recordCounter >= recordEvery:
            trajectoryCells.append(list(cells))
            trajectoryTimes.append(time)
            trajectoryBoundaries.append(max(0, N - l))
            trajectoryGenerations.append(generations)
            recordCounter = 0

    if trajectoryGenerations[-1] != generations:
        trajectoryCells.append(list(cells))
        trajectoryTimes.append(time)
        trajectoryBoundaries.append(max(0, N - l))
        trajectoryGenerations.append(generations)

    if terminationReason is None:
        terminationReason = 'unknown'

    extinct = (N == 0)

    # ----- Aggregate lineage statistics -----
    # Mutation supply: how many lineages ever appeared, by compartment of origin.
    nLineagesAppearedCore = sum(1 for info in lineages.values() if info['birthRegion'] == 'core')
    nLineagesAppearedEdge = sum(1 for info in lineages.values() if info['birthRegion'] == 'edge')

    # Fates: lineages that went extinct (liveCount hit 0) during the run.
    nLineagesExtinctCore = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'core' and info['liveCount'] == 0
    )
    nLineagesExtinctEdge = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and info['liveCount'] == 0
    )

    # Established lineages: ever reached maxLiveCount >= kEst (passes the drift bottleneck).
    nLineagesEstablished = sum(
        1 for info in lineages.values() if info['maxLiveCount'] >= kEst
    )
    nLineagesEstablishedCore = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'core' and info['maxLiveCount'] >= kEst
    )
    nLineagesEstablishedEdge = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and info['maxLiveCount'] >= kEst
    )

    # Phase 1 vs Phase 2 stratification of edge-born lineages
    def _isPhase1(info):
        return phase1EndTime is None or info['birthTime'] < phase1EndTime

    nLineagesAppearedEdgePhase1 = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and _isPhase1(info)
    )
    nLineagesAppearedEdgePhase2 = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and not _isPhase1(info)
    )
    nLineagesEstablishedEdgePhase1 = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and _isPhase1(info)
        and info['maxLiveCount'] >= kEst
    )
    nLineagesEstablishedEdgePhase2 = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and not _isPhase1(info)
        and info['maxLiveCount'] >= kEst
    )

    # Edge-born lineages the conveyor pushed into the core at least once, and the
    # total number of such sweeps. Only informative when R is costed in the core.
    nLineagesEdgeBornEnteredCore = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'edge' and info['everEnteredCore']
    )
    nSweepsEdgeBorn = sum(
        info['nSweptToCore'] for info in lineages.values()
        if info['birthRegion'] == 'edge'
    )

    # Core-born lineages that ever reached the edge (either by core->edge boundary
    # flow, or by edge birth after arrival -- either way, everReachedEdge is set).
    nLineagesReachedEdge = sum(
        1 for info in lineages.values()
        if info['birthRegion'] == 'core' and info['everReachedEdge']
    )

    # Mean lineage size at first edge entry for delivered core-born lineages.
    # deliverySize records total liveCount at the moment everReachedEdge first
    # became True. Larger values indicate multi-cell bolus arrivals.
    _delivered = [
        info['deliverySize'] for info in lineages.values()
        if info['birthRegion'] == 'core' and info['everReachedEdge']
    ]
    meanDeliverySizeCore = sum(_delivered) / len(_delivered) if _delivered else None

    # State at termination. Scan cells once to get per-lineage compartment counts.
    bFinal = max(0, N - l)
    livePresentLineages = {lin for lin, info in lineages.items() if info['liveCount'] > 0}
    lineagesInCore = set()
    lineagesInEdge = set()
    for i in range(bFinal):
        c = cells[i]
        if c >= 2:
            lineagesInCore.add(c)
    for i in range(bFinal, N):
        c = cells[i]
        if c >= 2:
            lineagesInEdge.add(c)

    nLineagesPresentAtEnd = len(livePresentLineages)
    nLineagesPresentInCore = len(lineagesInCore)
    nLineagesPresentInEdge = len(lineagesInEdge)

    # Rescue-specific stats
    # rescueEdgeCounts was captured at the moment rEdge first crossed rThreshold,
    # regardless of stopAtRescue. Using the snapshot gives correct lineage counts
    # at the rescue moment even when the simulation continues past rescue (in
    # which case the winning lineage would otherwise drive the others out).

    if rescued:
        nLineagesAtRescueEdge = len(rescueEdgeCounts)

        if rescueEdgeCounts:
            # Primary: highest count, tie-break by most recent birthTime.
            def primaryKey(item):
                lin, cnt = item
                return (cnt, lineages[lin]['birthTime'])
            primaryLineage, primaryLineageCount = max(rescueEdgeCounts.items(), key=primaryKey)
            primaryInfo = lineages[primaryLineage]
            primaryLineageOrigin = primaryInfo['birthRegion']
            primaryLineageBirthTime = primaryInfo['birthTime']
        else:
            # Shouldn't happen if rescued (rEdge >= rThreshold implies >= 1 R in edge).
            primaryLineage = None
            primaryLineageCount = 0
            primaryLineageOrigin = None
            primaryLineageBirthTime = None
    else:
        nLineagesAtRescueEdge = None
        primaryLineage = None
        primaryLineageCount = None
        primaryLineageOrigin = None
        primaryLineageBirthTime = None

    # rescueMode kept for back-compat, now equals primaryLineageOrigin.
    rescueMode = primaryLineageOrigin

    return {
        'rescued': rescued,
        'rescueTime': rescueTime,
        'rescueGeneration': rescueGeneration,
        'rescueMode': rescueMode,
        'extinct': extinct,
        'extinctionTime': time if extinct else None,
        'extinctionGeneration': generations if extinct else None,
        'terminationReason': terminationReason,
        'finalN': N,
        'finalTime': time,
        'finalGenerations': generations,
        'finalCells': cells,
        'mutationEvents': mutationEvents,
        # Aggregate lineage statistics
        'nLineagesAppeared': nLineagesAppearedCore + nLineagesAppearedEdge,
        'nLineagesAppearedCore': nLineagesAppearedCore,
        'nLineagesAppearedEdge': nLineagesAppearedEdge,
        'nLineagesExtinctCore': nLineagesExtinctCore,
        'nLineagesExtinctEdge': nLineagesExtinctEdge,
        'nLineagesReachedEdge': nLineagesReachedEdge,
        'meanDeliverySizeCore': meanDeliverySizeCore,
        'nLineagesEstablished': nLineagesEstablished,
        'nLineagesEstablishedCore': nLineagesEstablishedCore,
        'nLineagesEstablishedEdge': nLineagesEstablishedEdge,
        'nLineagesEdgeBornEnteredCore': nLineagesEdgeBornEnteredCore,
        'nSweepsEdgeBorn': nSweepsEdgeBorn,
        'l': l,
        'phase1EndTime': phase1EndTime,
        'nLineagesAppearedEdgePhase1': nLineagesAppearedEdgePhase1,
        'nLineagesAppearedEdgePhase2': nLineagesAppearedEdgePhase2,
        'nLineagesEstablishedEdgePhase1': nLineagesEstablishedEdgePhase1,
        'nLineagesEstablishedEdgePhase2': nLineagesEstablishedEdgePhase2,
        'wtExtinct': wtExtinct,
        'wtExtinctTime': wtExtinctTime,
        'nLineagesPresentAtEnd': nLineagesPresentAtEnd,
        'nLineagesPresentInCore': nLineagesPresentInCore,
        'nLineagesPresentInEdge': nLineagesPresentInEdge,
        'rCoreAtEnd': rCore,
        'rEdgeAtEnd': rEdge,
        # Rescue-specific
        'nLineagesAtRescueEdge': nLineagesAtRescueEdge,
        'rescueEdgeCounts': rescueEdgeCounts,
        'primaryLineage': primaryLineage,
        'primaryLineageCount': primaryLineageCount,
        'primaryLineageOrigin': primaryLineageOrigin,
        'primaryLineageBirthTime': primaryLineageBirthTime,
        # Trajectory (unchanged)
        'trajectoryCells': trajectoryCells,
        'trajectoryTimes': trajectoryTimes,
        'trajectoryBoundaries': trajectoryBoundaries,
        'trajectoryGenerations': trajectoryGenerations,
    }