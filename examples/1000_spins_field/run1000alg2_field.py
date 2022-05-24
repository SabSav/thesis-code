"""Testing MC and A1 with three spins"""
import sys, os
import numpy as np

# Add code directory to the python path
sys.path.append('code')
from alg2 import simulate as alg2

size = 1000
lT = 0.5
hT = 10
h = 0.7
J = 1


action_rates = np.array([
    2 * [0.7 if i % 2 == 0 else 0.1]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))


def case_lt_without_correlations():
    """Produce decorrelated samples of alg2 simulation at `T = 0.5`"""
    alg2(
        size=size, temperature=lT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=1000, length=1000000, frame_step=100,
        output=f'{dir_path}/a2-lT.json'  # low T algorithm 1
    )


def case_ht_with_correlations():
    """Produce correlated samples of alg2 simulation at `T = 0.5`"""
    alg2(
        size=size, temperature=hT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=10, length=10000, frame_step=1,
        output=f'{dir_path}/a2-hT.json'  # low T algorithm 1
    )


if __name__ == '__main__':
    case_lt_without_correlations()
    case_ht_with_correlations()
