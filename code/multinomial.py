"""Tests for the multinomial distribution"""
import numpy as np
import scipy as sp
import scipy.stats

def one_sample_test(probabilities, counts, size=1000):
    """Resampling test with one sample from a multinomial distribution

    The larger is the p-value of the test, the more likely that the sample comes
    from a given multinomial distribution.

    Args:
        probabilities (numpy array of floats): theoretical probabilities for categories
        counts (numpy array of integers): Sample of categories' counts
        size: Size of the permutation sampling
    Returns:
        P-value for the hypothesis that the sample comes from a given multinomial
            distribution
    """
    tot = np.sum(counts)
    drb = sp.stats.multinomial(tot, probabilities)
    obs = drb.pmf(counts)

    counter = 0
    for _ in range(size):
        if drb.pmf(np.random.multinomial(tot, probabilities)) <= obs:
            counter += 1
    return counter / size

def one_sample_chi_squared(probabilities, counts, size=1000):
    """Chi-squared resampling test with one sample from a multinomial distribution

    The larger is the p-value of the test, the more likely that the sample comes
    from a given multinomial distribution.

    Args:
        probabilities (numpy array of floats): theoretical probabilities for categories
        counts (numpy array of integers): Sample of categories' counts
        size: Size of the permutation sampling
    Returns:
        P-value for the hypothesis that the sample comes from a given multinomial
            distribution
    """
    tot = np.sum(counts)
    num = len(counts) # number of categories
    est = tot * probabilities
    stat = np.sum((counts - est)**2 / est)

    counter = 0
    for _ in range(size):
        smp = np.random.multinomial(tot, probabilities)
        if np.sum((smp - est)**2 / est) > stat:
            counter += 1
    return counter / size
