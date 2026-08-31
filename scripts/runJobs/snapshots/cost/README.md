Frozen copies of the simulator and batch driver, taken when the core-only
cost-of-resistance sweep was submitted (branch dose-dependent-l).

Same reasoning as snapshots/penetration/: HTCondor transfers input files when a
job *starts*, not when it is submitted, so a long queue picks up whatever is on
disk at start time. This sweep is the first to use --costR and the
nEdgeBornEnteredCore / nSweepsEdgeBorn columns; freezing keeps queued jobs
consistent with the runs that already finished.
