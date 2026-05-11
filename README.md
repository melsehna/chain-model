# chain-model

Stochastic simulation study of **evolutionary rescue in biofilms under
antibiotic treatment**. A 1D chain model with a drug-exposed edge and a
shielded core is compared against a well-mixed population of the same
initial size, across a range of doses and core-activity levels.

The central finding: the biofilm is a **double-edged sword**. With an
active interior, it rescues less often than the well-mixed reference
at low dose and more often at high dose; with a dormant interior, it
loses uniformly at every dose.

## Layout

| Path | What's there |
|---|---|
| `src/`           | Simulators (`chainModel.py`, `wellMixed.py`) and visualization (`kymograph.py`) |
| `scripts/`       | Plotting and analysis scripts for the writeup figures (`plot*.py`, `aggregateResults.py`) |
| `scripts/runJobs/` | HTCondor / OSG sweep submission (`gen*Jobs.py`, `*.sub`, `runBatch.{py,sh}`) |
| `figures/`       | Aggregated sweep CSVs (`main.csv`, `sensL.csv`, `sensCore.csv`, `sensMu.csv`) and rendered `panels/` |
| `writing/`       | LaTeX writeups: `writeup_v2.tex` (main, current), `narrative.tex` (informal)|

## Where to start

- **The story**: `writing/writeup_v2.tex` is the current writeup with the
  full narrative, figures, and parameter table.
- **The model**: `src/chainModel.py` (`simulateChain`) and
  `src/wellMixed.py` (`simulateWellMixed`).
- **A quick run**: `cd scripts/runJobs && PYTHONPATH=../../src python quickSweep.py`
  (3 conditions × 8 doses × 200 seeds, ~6 min).

## Common tasks

```bash
# Single simulation
python -c "import sys; sys.path.insert(0, 'src'); from chainModel import simulateChain; \
print(simulateChain({'mu': 1e-4, 'nInit': 1000, 'l': 100, \
  'bWtCore': 0.2, 'dWtCore': 0.2, 'bWtEdge': 1.0, 'dWtEdge': 1.5, \
  'bREdge': 1.0, 'dREdge': 0.6}, seed=0)['rescued'])"

# Re-render all writeup figures from the CSVs
for s in scripts/plot*.py; do python $s; done

# Compile the writeup (inside Apptainer)
cd writing && apptainer exec ~/rescue-model-v1.sif pdflatex -interaction=nonstopmode writeup_v2.tex
```

For the full OSG sweep workflow, see `scripts/runJobs/` (the
`gen*Jobs.py` scripts emit HTCondor argument files, and each `*.sub`
is submitted with `condor_submit`).
