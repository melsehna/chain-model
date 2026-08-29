Frozen copies of the simulator and batch driver, taken when the dose-dependent
penetration sweep was submitted (branch dose-dependent-l).

HTCondor transfers input files when a job *starts*, not when it is submitted, so
a long queue picks up whatever is on disk at start time. Submitting against a
snapshot rather than the live tree means later edits cannot change the meaning of
jobs that are still idle. Use this pattern for any submission that will sit in the
queue while the source is being edited.
