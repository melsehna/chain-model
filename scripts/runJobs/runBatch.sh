#!/bin/bash
# Wrapper that HTCondor calls.
#
# Arguments: condition dose seedStart seedCount mu bCoreRatio outFile [l]
# The optional 8th arg is the edge width l. Omit to use the condition default
# (100 for biofilm, 2000 for chainWM).
#
# Run inside +SingularityImage=/cvmfs/.../htc/rocky:9 (Python 3.9 + numpy 1.23).
# No per-job tarball needed.

set -e

CONDITION=$1
DOSE=$2
SEED_START=$3
SEED_COUNT=$4
MU=$5
B_CORE_RATIO=$6
OUT_FILE=$7
L_VAL=${8:-}

echo "Running condition=${CONDITION} dose=${DOSE} seedStart=${SEED_START} seedCount=${SEED_COUNT} l=${L_VAL:-default}"
echo "Python: $(which python3)"
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)"

L_FLAG=()
if [ -n "${L_VAL}" ]; then
    L_FLAG=(--l "${L_VAL}")
fi

python3 runBatch.py \
    --condition "${CONDITION}" \
    --dose "${DOSE}" \
    --seedStart "${SEED_START}" \
    --seedCount "${SEED_COUNT}" \
    --mu "${MU}" \
    --bCoreRatio "${B_CORE_RATIO}" \
    "${L_FLAG[@]}" \
    --out "${OUT_FILE}"

echo "Done: ${OUT_FILE} has $(wc -l < ${OUT_FILE}) lines"