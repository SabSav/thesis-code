from collections import OrderedDict

import numpy as np
import scipy
from scipy import special as sp
from collections import OrderedDict


def equilibrate_counts(dict_a, dict_b):
    total_keys = np.unique(list(dict_a.keys()) + list(dict_b.keys()))
    for key in total_keys:
        if key not in dict_a.keys():
            dict_a[key] = 0
        if key not in dict_b.keys():
            dict_b[key] = 0

    # RB: Just `sorted(dict_*.items())` -> non funziona
    dict_a = OrderedDict(sorted(dict_a.items()))
    dict_b = OrderedDict(sorted(dict_b.items()))

    return dict_a, dict_b


def one_sample_chi_squared(probabilities, counts):
    """Chi-squared test with one sample from a multinomial distribution

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
        if est[i] == 0 and counts[i] == 0:  # no counts means one less degree of freedom
            df -= 1
    stat = np.sum((counts - est) ** 2 / est)
    return sp.gammaincc(df / 2, stat / 2)


def two_sample_chi_squared(counts1, counts2):
    """Two-samples chi-squared statistics"""
    num = len(counts1)
    assert num == len(counts2)
    tot1 = np.sum(counts1)
    tot2 = np.sum(counts2)
    plus = counts1 + counts2

    f1 = np.sqrt(tot2 / tot1)
    f2 = 1 / f1
    df = len(counts1) - 1
    stat = 0.0
    for i in range(num):
        if plus[i] == 0:  # no counts mean one less degree of freedom (book - numPress)
            df -= 1
        else:
            stat += (f1 * counts1[i] - f2 * counts2[i]) ** 2 / plus[i]

    return sp.gammaincc(df / 2, stat / 2)


def one_sample_chi_squared_resampling(probabilities, counts, size=100000):
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
    num = len(counts)  # number of categories
    est = tot * probabilities
    stat = 0
    for i in range(len(est)):
        if est[i] != 0: stat += ((counts[i] - est[i]) ** 2) / est[i]

    counter = 0
    for _ in range(size):
        value = 0
        smp = np.random.multinomial(tot, probabilities)
        for i in range(len(est)):
            if est[i] != 0: value += ((smp[i] - est[i]) ** 2) / est[i]
        if value > stat: counter += 1
    return counter / size


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

    return sp.gammaincc((num - 1) / 2, stat)  # and here do not divide ratio by 2


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
    sample1 = np.array([i for i in sample1 if i != 0])  # delete the zero counts
    sample2 = np.array([i for i in sample2 if i != 0])  # delete the zero counts
    sample_tot = np.array(
        [i for i in sample_tot if i != 0])  # delete the zero counts: I'll do here and not before because
    # is easier for the pairwise sum
    class_sample1 = len(sample1)
    class_sample2 = len(sample2)
    class_sample_tot = len(sample_tot)

    stat = - np.log(1 / tot1 + 1 / tot2)

    stat -= (class_sample1 + class_sample2 - class_sample_tot - 1) * np.log(2 * np.pi)
    stat -= np.sum(np.log(sample1)) + np.sum(np.log(sample2)) - np.sum(np.log(sample_tot))

    return sp.gammaincc((class_sample_tot - 1) / 2, stat / 2)  # this test gives a NaN result



