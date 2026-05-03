# Biofilm rescue parameter sweeps for OSG / HTCondor

Scripts to run large parameter sweeps of the spatial chain model on
HTCondor (OSG or local pool).

## Three conditions

Each sweep runs three conditions, chosen to give both the scientific
comparison and an internal consistency check:

| Condition | Model | Description |
|---|---|---|
| `biofilm`   | chainModel.py (l=100)  | Core + edge structure (ℓ=100, N_init=1000). Main object of study. |
| `chainWM`   | chainModel.py (l=2000) | Chain model with l ≥ N_init so no core ever forms. Chain model's "well-mixed limit." |
| `wellMixed` | wellMixed.py           | Separately implemented well-mixed birth-death model. |

**`chainWM` vs `wellMixed` is a consistency check**, not a scientific comparison.
The two should give statistically equivalent results: the chain model with
l ≥ N_init is mathematically equivalent to a well-mixed population. Running
both catches implementation errors -- if they diverge at any dose, something
is wrong. `testEquivalence.py` validates this at small scale (1000 paired seeds).

**`biofilm` vs `chainWM`/`wellMixed` is the scientific comparison.** Does the
biofilm's spatial structure give it a rescue advantage? At what doses?

## Files

| File | Purpose |
|---|---|
| `chainModel.py`      | Full spatial chain model |
| `wellMixed.py`       | Independent well-mixed birth-death model |
| `runBatch.py`        | Runs one (condition, dose, seeds) batch; writes CSV |
| `runBatch.sh`        | HTCondor shell wrapper |
| `genSweepMainJobs.py`        | Main sweep job list |
| `genSensitivityCoreJobs.py`  | Core dormancy sensitivity |
| `genSensitivityMuJobs.py`    | Mutation rate sensitivity |
| `sweepMain.sub`               | HTCondor submit for main sweep |
| `sweepSensitivityCore.sub`    | Submit for core sensitivity |
| `sweepSensitivityMu.sub`      | Submit for mu sensitivity |
| `aggregate.py`       | Combines CSVs, produces summary figures |
| `quickSweep.py`      | Local 3-condition sanity check (~3 min) |
| `testEquivalence.py` | Verifies chainWM agrees with wellMixed |

## Preliminary findings

Running `quickSweep.py` (3 conditions × 8 doses × 200 seeds, ~3 min) shows:

- **Biofilm has higher rescue probability than well-mixed at every dose.**
  No crossover within the super-MIC regime. Consistent with Fruet et al. 2025.
- **Core-delivery pathway fraction rises with dose** in the biofilm,
  from ~15% at d=1.05 to ~30-35% at d>1.5, then plateaus.
- **chainWM and wellMixed agree** within statistical noise at every dose
  (paired differences consistent with zero).

Full OSG sweep (1000 seeds × 20 doses) will give tighter error bars.

## Parameters (rescaled so bWtEdge = 1)

- `nInit = 1000`
- `l = 100` for biofilm, `l = 2000` for chainWM, (no l for wellMixed)
- `bWtEdge = 1.0`, `dWtEdge = dose` (sweep variable; super-MIC is dose > 1)
- `bWtCore = dWtCore = 0.2` (core 5x slower than edge)
- `bREdge = 1.0`, `dREdge = 0.6`
- `bRCore = dRCore = 0.2`
- `mu = 1e-4` (main sweep); swept in sensitivity

Time in edge generations.

## Sweeps

### Main sweep
- 20 doses log-spaced in `s = dose - 1` from 0.05 to 2.0
- 3 conditions (biofilm, chainWM, wellMixed)
- 1000 seeds per (condition, dose), in 5 batches of 200
- **Total: 300 jobs**, ~6s compute each

### Core dormancy sensitivity (biofilm only)
- bCoreRatio ∈ {0.05, 0.1, 0.2}
- 15 doses, 600 seeds per point
- **Total: 135 jobs**

### Mutation rate sensitivity (all three conditions)
- mu ∈ {1e-6, 1e-5, 1e-4, 1e-3}
- 3 conditions, 15 doses, 600 seeds per point
- **Total: 540 jobs**

## Deployment steps

### 1. Transfer to OSG submit host

```bash
rsync -av sweep/ user@ap40.uw.osg-htc.org:biofilm-sweep/
```

### 2. On submit host

```bash
cd biofilm-sweep
python genSweepMainJobs.py
python genSensitivityCoreJobs.py
python genSensitivityMuJobs.py
mkdir -p logs results
```

### 3. Customize submit files

In each `.sub` file:
- Replace `PLACEHOLDER_PROJECT` with your OSG project name
- If using a Singularity image, add:
  ```
  +SingularityImage = "/cvmfs/singularity.opensciencegrid.org/opensciencegrid/osgvo-el8:latest"
  ```
- Or uncomment `pip install --user numpy` in `runBatch.sh`

### 4. Submit

```bash
condor_submit sweepMain.sub             # 300 jobs
condor_submit sweepSensitivityCore.sub  # 135 jobs
condor_submit sweepSensitivityMu.sub    # 540 jobs
```

### 5. Aggregate

```bash
python aggregate.py --prefix main     --out main.png
python aggregate.py --prefix sensCore --out sensCore.png
python aggregate.py --prefix sensMu   --out sensMu.png
```

`main.png` has 3 panels: rescue vs dose for all 3 conditions, biofilm
pathway breakdown, and the chainWM-vs-wellMixed agreement check.
Watch the agreement-check panel: it should scatter around zero with
error bars crossing zero at every dose. Visible systematic deviation
is a red flag.

## Local testing

Single batch:
```bash
./runBatch.sh biofilm 1.5 0 50 1e-4 0.2 test.csv
```

Full quick sweep:
```bash
python quickSweep.py
```

Equivalence test (1000 paired seeds):
```bash
python testEquivalence.py
```