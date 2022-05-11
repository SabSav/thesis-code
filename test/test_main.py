import numpy as np
import pytest
# from ising import main

# def test_same_distribution():
#     random_seed = 1
#     np.random.seed(random_seed)
#     chain_size = 3
#     time_step = 1500
#     J = 1
#     h = 0
#     T = 10
#     n_samples = 1000
#     n_steps = 1000
#     t_burn_in = 10
#     t_correlation = 1
#     action_rates_plus = []
#     action_rates_minus = []
#     action_rate_values = [0.1, 0.7]
#     initial_config = 2 * np.random.randint(2, size=chain_size) - 1
#
#     for i in range(chain_size):
#         for j in range(len(action_rate_values)):
#             action_rates_plus.append(action_rate_values[j])
#             action_rates_minus.append(action_rate_values[j])
#
#     H_MMC, M_MMC = main.metropolis_ising(chain_size=chain_size, J=J, h=h, beta=1 / T, burn_in=t_burn_in,
#                                          correlation_time=t_correlation, n_samples=n_samples)
#     HAlg1, MAlg1 = main.dynamic_evaluation(chain_size=chain_size, T=T, J=J, h=h, n_steps=n_steps, config=initial_config,
#                                            action_rates_plus=action_rates_plus, action_rates_minus=action_rates_minus)
#
#     theory_H, theory_M, theory_H_counts = main.theory_en_m(h, J, T, chain_size, n_samples)
#     pvalue_H = main.chi2test(H_MMC, HAlg1, energy_flag=True)
#     pvalue_M = main.chi2test(M_MMC, MAlg1, energy_flag=False)
#
#     assert (pvalue_H, pvalue_M) == pytest.approx((0.7, 0.7), abs=3e-1)
#
#     # we can see the two distributions obtained from the different algorithms come from the same distribution

"""
def test_counts():
    random_seed = 1
    np.random.seed(random_seed)
    chain_size = 3
    time_step = 1500
    J = 1
    h = 0
    T = 10
    n_samples = 1000
    n_steps = 1000
    t_burn_in = 10
    t_correlation = 1
    action_rates_plus = []
    action_rates_minus = []
    action_rate_values = [0.1, 0.7]
    initial_config = 2 * np.random.randint(2, size=chain_size) - 1

    for i in range(chain_size):
        for j in range(len(action_rate_values)):
            action_rates_plus.append(action_rate_values[j])
            action_rates_minus.append(action_rate_values[j])

    H_MMC, M_MMC = main.metropolis_ising(chain_size=chain_size, J=J, h=h, beta=1 / T, burn_in=t_burn_in,
                                         correlation_time=t_correlation, n_samples=n_samples)
    HAlg1, MAlg1 = main.dynamic_evaluation(chain_size=chain_size, T=T, J=J, h=h, n_steps=n_steps, config=initial_config,
                                           action_rates_plus=action_rates_plus, action_rates_minus=action_rates_minus)

    theory_H, theory_M, theory_H_counts = main.theory_en_m(h, J, T, chain_size, n_samples)
    counts_HMMC = main.count_variables(H_MMC, energy_flag=True)
    counts_HAlg1 = main.count_variables(HAlg1, energy_flag=True)
    std_MMC = main.testing(H_MMC, T, n_samples,  list(counts_HMMC.values()))
    std_Alg1 = main.testing(HAlg1, T, n_samples,  list(counts_HAlg1.values()))
    std = main.testing(theory_H, T, n_samples, theory_H_counts)

"""
