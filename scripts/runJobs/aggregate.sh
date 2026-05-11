#!/bin/bash
# HTCondor wrapper for sweep aggregation + plotting.
# Runs inside the rescue-model-v1.sif image (numpy + pandas + matplotlib).
#
# Inputs (delivered by HTCondor):
#   sweepResults.tar.gz       — packaged scripts/runJobs/results/
#   aggregateResults.py       — per-prefix combined CSVs + summary PNG
#   plotSimPanels.py          — multi-panel narrative figures
#   plotTheory.py             — imported by plotSimPanels.py
#
# Output:
#   aggregateOutputs.tar.gz   — combined CSVs + all PNGs

set -e

echo "=== environment ==="
python3 --version
python3 -c "import numpy, pandas, matplotlib; print('numpy', numpy.__version__, 'pandas', pandas.__version__, 'matplotlib', matplotlib.__version__)"
echo

echo "=== unpack input ==="
tar xzf sweepResults.tar.gz
ls results/ | wc -l
echo "results CSVs unpacked"
echo

mkdir -p outputs/panels

echo "=== concatenate per-prefix CSVs (needed by plotSimPanels) ==="
python3 <<'PY'
import pandas as pd
import glob

for prefix in ['main', 'sensCore', 'sensMu', 'sensL']:
    files = sorted(glob.glob(f'results/{prefix}_*.csv'))
    if not files:
        print(f'  {prefix}: no files (skipping)')
        continue
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = f'outputs/{prefix}.csv'
    df.to_csv(out, index=False)
    print(f'  {prefix}: {len(files)} files -> {out} ({len(df):,} rows)')
PY

echo
echo "=== per-prefix summary figures (aggregateResults.py) ==="
for prefix in main sensCore sensMu sensL; do
    [ -f outputs/${prefix}.csv ] || continue
    python3 aggregateResults.py --resultsDir results --prefix ${prefix} --out outputs/${prefix}.png
done

echo
echo "=== multi-panel narrative figures (plotSimPanels.py) ==="
python3 plotSimPanels.py --resultsDir outputs --outDir outputs/panels

echo
echo "=== outputs ==="
ls -la outputs/
ls -la outputs/panels/

echo
echo "=== package outputs ==="
tar czf aggregateOutputs.tar.gz outputs/
ls -lh aggregateOutputs.tar.gz
