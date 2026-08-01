"""Does the boundary correction agree with an independent estimator?

Everything gated so far says the correction is well-FORMED: forward value exactly unchanged, the
counterfactual replay reproduces the reported MTM, the term lands only on factors that enter the
decision gap, and it converges as bandwidth shrinks and paths grow. None of that says the NUMBER
is right — a consistently-estimated wrong quantity would pass every one of those.

Bump-and-reprice is the wrong production architecture, and the right oracle. Under common random
numbers - same seed, so the same normals, since the draws depend on the seed and factor ordering
and not on the spot value - a central difference of CVA in the EUR spot is a direct estimate of
the same derivative AAD reports, and it is blind to how AAD got there. Three numbers then say
everything:

    AAD only            the frozen-decision gradient riskflow reports today
    AAD + correction    what this branch reports with Boundary_AAD on
    CRN central diff    the oracle

If CRN sits on the corrected value, the correction is right and the shipped gradient is wrong by
the difference. If CRN sits on the uncorrected one, the correction is wrong however nicely it
converges.

A ladder of bumps rather than one: too small and the difference drowns in what CRN does not
cancel, too large and it measures curvature instead of the derivative. The reading is only
trustworthy where the ladder is flat.

Run:  CUDA_VISIBLE_DEVICES=0 python gates/boundary_aad_crn.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tests'))

import numpy as np
import torch

import riskflow
import test_boundary_aad_mta as fixture

MTA = 2_000_000.0
SPOT = 1.1
SEED = 1
PATHS = 32768
BANDWIDTH = 0.02


def cva(spot, batch=PATHS, seed=SEED):
    """CVA with EUR spot moved, everything else - crucially the seed - held."""
    cfg = fixture._cfg(MTA)
    cfg.params['Price Factors']['FxRate.EUR']['Spot'] = spot
    _, out = riskflow.run_cmc(cfg, prec=fixture.DTYPE,
                              overrides=fixture._params(False, seed=seed, batch=batch))
    return float(out['Results']['cva'])


def aad_delta(boundary_aad, batch=PATHS, seed=SEED, bandwidth=BANDWIDTH):
    params = fixture._params(boundary_aad, seed=seed, batch=batch)
    if boundary_aad:
        params['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = riskflow.run_cmc(fixture._cfg(MTA), prec=fixture.DTYPE, overrides=params)
    grad = out['Results']['grad_cva']['Gradient']
    return float(grad.iloc[0])                       # FxRate.EUR


print(f'MTA {MTA:,.0f}   spot {SPOT}   paths {PATHS:,}   seed {SEED}   bandwidth {BANDWIDTH}\n')
plain = aad_delta(False)
corrected = aad_delta(True)
print(f'  AAD only          {plain:14,.0f}')
print(f'  AAD + correction  {corrected:14,.0f}    (correction {corrected - plain:+,.0f})\n')

print(f'  {"rel bump":>10} {"CRN delta":>14} {"vs AAD":>10} {"vs corrected":>14}')
for rel in (1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3):
    h = SPOT * rel
    crn = (cva(SPOT + h) - cva(SPOT - h)) / (2.0 * h)
    print(f'  {rel:10.1e} {crn:14,.0f} {abs(crn - plain) / max(abs(plain), 1e-30):9.1%} '
          f'{abs(crn - corrected) / max(abs(corrected), 1e-30):13.1%}')
