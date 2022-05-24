"""Testing Alg2 with three spins"""
import sys, os
import numpy as np

# Add code directory to the python path
sys.path.append('code')
from alg2 import simulate as alg2

size = 3
lT = 0.5
hT = 10
h = 0
J = 1

action_rates = np.array([
    2 * [0.7 if i % 2 == 0 else 0.1]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))


def case_lt_alg2_without_correlations():
    """Produce decorrelated samples of alg1 simulation at `T = 0.5`"""

    alg2(
        size=size, temperature=lT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=100000, length=10000000, frame_step=100,
        output=f'{dir_path}/a2-lT.json'  # low T algorithm 1
    )


def case_ht_alg2_without_correlations():
    """Produce decorrelated samples of alg1 simulation at `T = 10`"""
    alg2(
        size=size, temperature=hT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=100000, length=10000000, frame_step=100,
        output=f'{dir_path}/a2-hT.json'  # high  T algorithm 1
    )


if __name__ == '__main__':

    case_lt_alg2_without_correlations()
    case_ht_alg2_without_correlations()

