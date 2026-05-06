#!/bin/bash
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/sweep_%x_%j.out
#SBATCH --error=logs/sweep_%x_%j.err
#
# Submit one runSweepLocal phase sweep on this node via Slurm.
#
# Usage:
#   sbatch --job-name=phaseL  --export=ALL,SWEEP=phaseL  scripts/slurm/runSweep.sbatch
#   sbatch --job-name=phaseMu --export=ALL,SWEEP=phaseMu scripts/slurm/runSweep.sbatch
#
# Optional: chain phaseMu after phaseL via dependency:
#   J1=$(sbatch --parsable --job-name=phaseL  --export=ALL,SWEEP=phaseL  scripts/slurm/runSweep.sbatch)
#   sbatch --dependency=afterok:$J1 --job-name=phaseMu --export=ALL,SWEEP=phaseMu scripts/slurm/runSweep.sbatch
#
# SEEDS env var overrides default 2500 seeds/point.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

SWEEP="${SWEEP:-phaseBc}"
SEEDS="${SEEDS:-2500}"

echo "=== Slurm sweep ==="
echo "Sweep:        $SWEEP"
echo "Seeds/point:  $SEEDS"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Node:         $(hostname)"
echo "Submit dir:   $SLURM_SUBMIT_DIR"
echo "Job ID:       $SLURM_JOB_ID"
echo "Started:      $(date)"
echo

/usr/bin/python3 scripts/runSweepLocal.py \
    --sweep "$SWEEP" \
    --seedsPerPoint "$SEEDS" \
    --nWorkers "$SLURM_CPUS_PER_TASK"

echo
echo "Finished: $(date)"
