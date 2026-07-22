"""Suite-wide test hygiene.

Pin the global torch default dtype to float32 at the start of EVERY test. Once enough test modules
are collected in one session the process-wide default flips to float64 at COLLECTION time — a
pre-existing interaction (12+ modules trip it; removing any single module drops it back, so it is a
threshold effect on the imported set, not one culprit, and it is orthogonal to what any test
asserts). A value-function net then built under a float64 default multiplies a float32 input and
dies with "mat1 and mat2 must have the same dtype", failing the solver / GARCH-generate / bit-exact
tests for a reason unrelated to their subject — while each passes in isolation.

Resetting per test makes the full suite green (verified) and is standard torch-test hygiene. Tests
that deliberately need float64 (test_symlog_unit's exact FD gates) set it inside their own body,
AFTER this fixture's setup, so they are unaffected.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _pin_default_dtype():
    torch.set_default_dtype(torch.float32)
    yield
