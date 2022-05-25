"""Two-samples statistical tests"""
import numpy as np
import scipy as sp
import scipy.stats


def chi_squared_stat(counts1, counts2):
    """Two-samples chi-squared statistics"""
    num = len(counts1)
    assert num == len(counts2)
    tot1 = np.sum(counts1)
    tot2 = np.sum(counts2)
    plus = counts1 + counts2

    f1 = np.sqrt(tot2 / tot1)
    f2 = 1 / f1
    df = len(counts1) - 1
    chi2 = 0.0
    for i in range(num):
        if plus[i] == 0:  # no counts mean one less degree of freedom (book - numPress)
            df -= 1
        else:
            chi2 += (f1 * counts1[i] - f2 * counts2[i]) ** 2 / plus[i]
    return df, chi2


def chi_squared_test(counts1, counts2):
    """Apply two-samples chi-squared test"""
    return sp.special.gammaincc(chi_squared_stat(counts1, counts2)[0] / 2,
                                chi_squared_stat(counts1, counts2)[1] / 2
                                )


def chi_squared_permutation_test(counts1, counts2, size):
    """Chi-squared permutation test with two samples from a multinomial
    distribution

    The larger is the p-value of the test, the more likely that the samples come
    from the same multinomial distribution.

    Args:
        counts1 (int[]): Number of events in the first sample
        counts2 (int[]): Number of events in the second sample
        size (int): Number of permutations to sample
    Returns:
        P-value for the hypothesis that the sample comes from a given multinomial
            distribution
    """
    obs = chi_squared_stat(counts1, counts2)
    tot1 = np.sum(counts1)
    tot2 = np.sum(counts2)

    events = np.empty(tot1 + tot2)
    index = 0
    offset = 0
    for k in counts1 + counts2:
        next = offset + k
        events[offset:next] = index
        index += 1
        offset = next

    counter = 0
    bins = np.arange(float(len(counts1)) + 1) - 0.5
    for _ in range(size):
        permutation = events[np.random.permutation(len(events))]
        chi2 = chi_squared_stat(
            np.histogram(permutation[:tot1], bins=bins)[0],
            np.histogram(permutation[tot1:], bins=bins)[0]
        )
        if chi2 > obs: counter += 1
    return counter / size
