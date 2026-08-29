'''Generate args for the dose-dependent penetration depth experiment.

A fixed edge width assumes the kill zone has the same thickness at 1.05x MIC as
at 2.8x MIC, which cannot happen. With attenuation c(z) = c0*exp(-z/lambda), the
depth above MIC is

    l(dose) = lambda * ln(dose / MIC),      MIC at dWtEdge = bWtEdge

so the refuge shrinks as the dose rises and is abolished entirely at
dose = exp(N0/lambda). Everything downstream (T1, theta, the capacity C) then
inherits a second, opposing dose dependence.

Two penetration lengths, both with N0 = 1000:
  lambda = 100   l = 5 at 1.05x MIC, 103 at 2.8x; refuge never abolished in range
  lambda = 300   l = 15 at 1.05x MIC, 309 at 2.8x; refuge abolished at ~28x MIC

Active (b_c = 0.2) and dormant (b_c = 0) cores. The well-mixed reference is
chainWM from the fxMain sweep, which is unaffected by lambda; matched settings
(mu = 1e-4, K = 1000, densityForm = step, kEst = 8).

Columns: dose seedStart bCore lambdaPen tag
'''

import itertools

doses         = [1.05, 1.1, 1.2, 1.4, 1.6, 1.9, 2.3, 2.8]
lambdas       = [100, 300]
bcores        = [0.2, 0.0]
seeds_total   = 2000
seeds_per_job = 200

rows = []
for lam, bc, dose, s in itertools.product(lambdas, bcores, doses,
                                          range(0, seeds_total, seeds_per_job)):
    rows.append((dose, s, bc, lam, f'lam{lam}bc{bc}'))

with open('penetrationJobs.txt', 'w') as f:
    for dose, s, bc, lam, tag in rows:
        f.write(f'{dose} {s} {bc} {lam} {tag}\n')
print(f'penetrationJobs.txt: {len(rows)} jobs '
      f'({len(lambdas)} lambdas x {len(bcores)} cores x {len(doses)} doses x '
      f'{seeds_total // seeds_per_job} batches), {seeds_total} seeds per point')
