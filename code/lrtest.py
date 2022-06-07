"""Likelihood ratio tests for multinomiial distributions"""
import numpy as np
import scipy as sp
from scipy import special


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
    num = len(counts)  # number of categories
    # Set 1 for absent counts, for their contribution will be ignored
    est = np.array([(count if count > 0 else 1) for count in counts]) / tot
    stat = -np.dot(counts, np.log(probabilities / est))  # no need to multiply ratio by 2

    stat /= 1 + (
            np.sum(1.0 / probabilities) - 1
    ) / (6 * tot * (num - 1))  # Williams' correction

    return sp.special.gammaincc((num - 1) / 2, stat)  # and here do not divide ratio by 2

    # # Smith method simplified to O(1/N^2) does not improve the situation so much
    # stat = -2 * np.dot(counts, np.log(probabilities / est))
    # parM = -2 * tot * np.log(np.min(probabilities))
    # pinv = 1 / probabilities
    # parE = num - 1 + (np.sum(pinv) - 1) / (6 * tot)
    # parV = 2 * (num - 1) + (np.sum(pinv) - 1) / (2 * tot / 3)
    #
    # a = parE / (parM * parV) * (parE * (parM - parE) - parV)
    # b = (parM - parE) * a / parE
    # return 1 - sp.special.betainc(a, b, stat / parM)


def one_sample_chi_squared(probabilities, counts):
    """Chi-squred test with one sample from a multinomial distribution

    The larger is the p-value of the test, the more likely that the sample comes
    from a given multinomial distribution.

    Args:
        probabilities (numpy array of floats): theoretical probabilities for categories
        counts (numpy array of integers): Sample of categories' counts
    Returns:
        P-value fr the hypothesis that the sample comes from a given multinomial
            distribution
    """
    tot = np.sum(counts)
    num = len(counts)  # number of categories
    df = num - 1
    est = tot * probabilities
    for i in range(num):
        if est[i] == 0 and counts[i] == 0:  # no counts mean one less degree of freedom
            df -= 1
    stat = np.sum((counts - est) ** 2 / est)
    return sp.special.gammaincc(df / 2, stat / 2)


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
    sample_tot = sample1 + sample2  # merge the two sample
    sample1 = np.array([i for i in sample1 if i!=0]) #delete the zero counts
    sample2 = np.array([i for i in sample2 if i!=0]) #delete the zero counts
    sample_tot = np.array([i for i in sample_tot if i!=0]) #delete the zero counts: i'll do here and not before because
    #is easerier for the pairwise sum
    class_sample1 = len(sample1)
    class_sample2 = len(sample2)
    class_sample_tot = len(sample_tot)

    if tot1 != tot2:
        stat = np.log(1 / tot1 + 1 / tot2)
    else:
        stat = np.log(2 / tot1)

    stat += (class_sample1 + class_sample2 - class_sample_tot - 1) * np.log(2 * np.pi)
    stat += np.sum(np.log(sample1)) + np.sum(np.log(sample2)) - np.sum(np.log(sample_tot))
    stat = stat / 2

    return sp.special.gammaincc((class_sample_tot - 1) / 2, stat / 2)
