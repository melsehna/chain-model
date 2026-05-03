#!/bin/bash
# Wrapper that HTCondor calls.
#
# Arguments: condition dose seedStart seedCount mu bCoreRatio outFile

set -e

CONDITION=$1
DOSE=$2
SEED_START=$3
SEED_COUNT=$4
MU=$5
B_CORE_RATIO=$6
OUT_FILE=$7

# If using OSG without a Singularity image, install numpy per job:
# python3 -m pip install --user numpy

echo "Running condition=${CONDITION} dose=${DOSE} seedStart=${SEED_START} seedCount=${SEED_COUNT}"
echo "Python: $(which python3)"
python3 --version

python3 runBatch.py \
    --condition "${CONDITION}" \
    --dose "${DOSE}" \
    --seedStart "${SEED_START}" \
    --seedCount "${SEED_COUNT}" \
    --mu "${MU}" \
    --bCoreRatio "${B_CORE_RATIO}" \
    --out "${OUT_FILE}"

echo "Done: ${OUT_FILE} has $(wc -l < ${OUT_FILE}) lines"