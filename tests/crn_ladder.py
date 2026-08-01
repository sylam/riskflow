"""Is a reported gradient the derivative of the value actually reported?

Under common random numbers - same seed, so the same normals, the draws depending on seed and
factor ordering rather than on the value being bumped - a central difference estimates the same
derivative AAD reports, arrived at without touching the tape.

Agreement at ONE bump size proves little. Too small and the difference drowns in what CRN does not
cancel; too large and it measures curvature. The reading is only trustworthy where the ladder is
FLAT, and flatness is the discriminating signal in its own right: differencing across a genuine
discontinuity produces readings that scatter with h and do not converge, because shrinking h
changes how many paths sit on the wrong side of the jump rather than refining a limit. Every gap
this measured in riskflow showed exactly that signature - a scattering ladder beside a stable AAD
number that looked perfectly well-behaved on its own.

So this reports three things and refuses to collapse them into one:
    agreement  - |crn - aad| / |aad| at each rung
    flatness   - spread of the CRN readings across rungs, relative to their median
    verdict    - agreement is only meaningful where flatness holds

Usage - the caller closes over its own config, so nothing here knows about riskflow:

    from crn_ladder import ladder
    r = ladder(price=lambda s: cva(spot=s), aad=aad_delta(), base=100.0)
    print(r)
    assert r.agrees(tol=0.02), str(r)
"""
import numpy as np

DEFAULT_RUNGS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)


class Ladder(object):
    """The result of comparing an AAD gradient against a CRN bump ladder."""

    def __init__(self, aad, base, rungs, crn):
        self.aad, self.base, self.rungs, self.crn = aad, base, np.asarray(rungs), np.asarray(crn)

    @property
    def agreement(self):
        """|crn - aad| / |aad| at each rung."""
        return np.abs(self.crn - self.aad) / max(abs(self.aad), 1e-30)

    @property
    def flatness(self):
        """Spread of the CRN readings relative to their median. Large means the ladder is not
        converging on anything, so no single rung is worth quoting."""
        med = np.median(self.crn)
        return float(np.ptp(self.crn) / max(abs(med), 1e-30))

    @property
    def best(self):
        """The CRN reading at the rung where the ladder is locally flattest - the one to quote."""
        if len(self.crn) < 3:
            return float(np.median(self.crn))
        # local curvature of the ladder; the flattest interior point is the stable read
        wobble = np.abs(np.diff(self.crn, 2))
        return float(self.crn[1 + int(wobble.argmin())])

    def agrees(self, tol=0.02, flat_tol=0.10):
        """AAD is the derivative of the reported value: the ladder converges AND lands on it."""
        return self.flatness <= flat_tol and abs(self.best - self.aad) / max(abs(self.aad), 1e-30) <= tol

    def __str__(self):
        head = (f'AAD {self.aad:+.6g}   CRN(best) {self.best:+.6g}   '
                f'disagreement {abs(self.best - self.aad) / max(abs(self.aad), 1e-30):.2%}   '
                f'flatness {self.flatness:.2%}'
                f'{"" if self.flatness <= 0.10 else "  <-- NOT FLAT, the CRN reading is not converging"}')
        rows = '\n'.join(f'    {h:9.1e} {c:+16.8g} {a:9.2%}'
                         for h, c, a in zip(self.rungs, self.crn, self.agreement))
        return f'{head}\n    {"rel bump":>9} {"CRN central diff":>16} {"vs AAD":>9}\n{rows}'


def ladder(price, aad, base, rungs=DEFAULT_RUNGS, absolute=False):
    """Central-difference `price` about `base` at each rung and compare against `aad`.

    price    callable taking the bumped value, returning the scalar AAD differentiates
    aad      the gradient the engine reported, d(scalar)/d(base)
    base     the unbumped value of the input being differentiated
    rungs    bump sizes, RELATIVE to base unless `absolute`
    """
    crn = []
    for h in rungs:
        step = h if absolute else base * h
        crn.append((price(base + step) - price(base - step)) / (2.0 * step))
    return Ladder(aad, base, rungs, crn)
