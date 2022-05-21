"""Test suit for `lrtest` module"""
from pytest import *
import lrtest
import numpy as np
import scipy as sp

def test_one_sample_test():
    """Test `one_sample_test()`"""
    size = 100000
    n = 10

    p = 0.1
    q = 1 - p
    np.random.seed(0)
    sample = np.random.binomial(n, p, size=size)
    bins = np.arange(float(n)+2)-0.5
    counts, _ = np.histogram(sample, bins)

    theory = np.array([sp.special.binom(n, k) * p**k * q**(n - k) for k in range(n+1)])
    assert lrtest.one_sample_test(theory, counts) > 0.99

    p = 0.9
    q = 1 - p
    theory = np.array([sp.special.binom(n, k) * p**k * q**(n - k) for k in range(n+1)])
    assert lrtest.one_sample_test(theory, counts) < 0.01
