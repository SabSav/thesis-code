"""Likelihood ratio tests for multinomiial distributions"""
import numpy as np
import scipy as sp

def one_sample_test(probabilities, counts):
    """Likelihood ratio test with one sample from a multinomial distribution

    The larger is the p-value of the test, the more likely that the sample comes
    from a given multinomial distribution.

    Args:
        probabilities (numpy array of floats): theoretical probabilities for categories
        counts (numpy array of integers): Sample of categories' counts
    Returns:
        P-value for the hypothesis that the sample comes from a given multinomial
            distribution
    """
    tot = np.sum(counts)
    # Set 1 for absent counts, for their contibution will be ignored
    est = np.array([(count if count > 0 else 1) for count in counts]) / tot
    ratio = -2.0 * np.dot(counts, np.log(probabilities / est))
    return sp.special.gammaincc((tot - 1) / 2, ratio / 2)

def two_samples_test(sample1, sample2):
    """Likelihood ratio test with two samples from a multinomial distribution

    The larger is the p-value of the test, the more likely that the samples come
    from the same multinomial distribution.

    Args:
        sample1 (numpy array of floats): Sample of categories' counts
        sample2 (numpy array of floats): Sample of categories' counts
    Returns:
        P-value for the hypothesis that the sample comes from a given multinomial
            distribution
    """
    tot1 = np.sum(sample1)
    tot2 = np.sum(sample2)
    # Set 1 for absent counts, for their contibution will be ignored
    probabilities = np.array(
        [(count if count > 0 else 1) for count in counts]
    ) / tot
    est = np.array([(count if count > 0 else 1) for count in counts]) / tot

    pass
