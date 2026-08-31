'''Is the shielded interior a graveyard for surface-born mutants?

Mutation is switched off inside (muCore = 0) while the interior stays active, so:
  - no interior-born mutants exist, hence no extra mutants delivered to the surface
  - no interior-born mutant can win the race and stop the run early, so surface-born
    mutants are no longer cut off mid-growth by someone else's success

What remains is only this: the boundary slides, so surface-born mutants get pushed
into the interior, where resistance buys them nothing and a busy interior can kill
them by chance. Everything else matches a well-mixed population.

Comparison that matters is against the dormant interior (b_c = 0), which does the
same sliding but freezes whatever it swallows instead of churning it. Same geometry,
same supply, different interior. Any gap between them is the graveyard effect.

Columns: dose seedStart bCore tag
'''
import itertools

seeds_total, seeds_per_job = 5000, 200
batches = range(0, seeds_total, seeds_per_job)

rows = []
for bc, dose, s in itertools.product([0.8, 0.2], [1.05, 1.2, 1.6], batches):
    rows.append((dose, s, bc, f'bc{bc:g}nomu'))
for s in batches:                       # dormant control, no churn to speak of
    rows.append((1.05, s, 0.0, 'bc0nomu'))

with open('sinkJobs.txt', 'w') as f:
    for dose, s, bc, tag in rows:
        f.write(f'{dose} {s} {bc} {tag}\n')
print(f'sinkJobs.txt: {len(rows)} jobs, {seeds_total} seeds per point')
