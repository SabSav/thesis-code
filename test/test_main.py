import pytest
import main
import numpy as np


def test_same_distribution():
    random_seed = 1
    np.random.seed(random_seed)
    chain_size = 3
    J = 1
    h = 0
    T = 0.5
    n_samples = 1000
    n_steps = 1000
    t_burn_in = 10
    t_correlation = 1
    # we define action rate plus and minus because the action rates could depend on the fact that the spin is +/-1
    action_rates_plus = []
    action_rates_minus = []
    action_rate_values = [0.1, 0.7]
    separator = " "
    string_action_rate = separator.join([str(_) for _ in action_rate_values])
    for i in range(chain_size):
        for j in range(len(action_rate_values)):
            action_rates_plus.append(action_rate_values[j])
            action_rates_minus.append(action_rate_values[j])
    initial_config = 2 * np.random.randint(2, size=chain_size) - 1

    energy_MMC, magn_MMC = main.metropolis_ising(chain_size=chain_size, J=J, h=h, beta=1 / T, burn_in=t_burn_in,
                                                 correlation_time=t_correlation, n_samples=n_samples)
    energy_Alg1, magn_Alg1 = main.dynamic_evaluation(chain_size=chain_size, J=J, h=h, T=T, n_steps=n_steps,
                                                     config=initial_config, action_rates_plus=action_rates_plus,
                                                     action_rates_minus=action_rates_minus)
    counts_HMMC, counts_HAlg1, bins_bound = main.centered_histogram(energy_MMC, energy_Alg1, J, chain_size,
                                                                    energy_flag=True)

    counts_MMMC, counts_MAlg1, bins_bound = main.centered_histogram(magn_MMC, magn_Alg1, J, chain_size,
                                                                    energy_flag=False)
    pvalue_H = main.chi2test(energy_MMC, energy_Alg1, counts_HMMC, counts_HAlg1)
    pvalue_M = main.chi2test(magn_MMC, magn_MMC, counts_MMMC, counts_MAlg1)

    assert (pvalue_H, pvalue_M) == pytest.approx((0.9, 0.9), abs=7e-2)
    #we can see the two distributions obtained from the different algorithms come from the same distribution

def test_histogram():
    random_seed = 1
    np.random.seed(random_seed)
    chain_size = 3
    J = 1
    h = 0
    T = 0.5
    n_samples = 1000
    n_steps = 1000
    t_burn_in = 10
    t_correlation = 1
    # we define action rate plus and minus because the action rates could depend on the fact that the spin is +/-1
    action_rates_plus = []
    action_rates_minus = []
    action_rate_values = [0.1, 0.7]
    separator = " "
    string_action_rate = separator.join([str(_) for _ in action_rate_values])
    for i in range(chain_size):
        for j in range(len(action_rate_values)):
            action_rates_plus.append(action_rate_values[j])
            action_rates_minus.append(action_rate_values[j])
    initial_config = 2 * np.random.randint(2, size=chain_size) - 1

    energy_MMC, magn_MMC = main.metropolis_ising(chain_size=chain_size, J=J, h=h, beta=1 / T, burn_in=t_burn_in,
                                                 correlation_time=t_correlation, n_samples=n_samples)
    energy_Alg1, magn_Alg1 = main.dynamic_evaluation(chain_size=chain_size, J=J, h=h, T=T, n_steps=n_steps,
                                                     config=initial_config, action_rates_plus=action_rates_plus,
                                                     action_rates_minus=action_rates_minus)
    counts_HMMC, counts_HAlg1, bins_bound = main.centered_histogram(energy_MMC, energy_Alg1, J, chain_size,
                                                                    energy_flag=True)

    binom_avg_MMC, binom_std_MMC = main.testing(energy_MMC, T, n_samples, counts_HMMC)
    binom_avg_Alg1, binom_std_Alg1 = main.testing(energy_Alg1, T, n_samples, counts_HAlg1)

    for i in range(len(counts_HMMC)):
        assert counts_HMMC[i] == pytest.approx(binom_avg_MMC[i], binom_std_MMC[i])

    for i in range(len(counts_HAlg1)):
        assert counts_HAlg1[i] == pytest.approx(binom_avg_Alg1[i], binom_std_Alg1[i])

