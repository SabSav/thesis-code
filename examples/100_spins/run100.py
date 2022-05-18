"""Testing MC and A1 with three spins"""
import sys, os
import numpy as np
import datetime

# Add code directory to the python path
sys.path.append('code')
from mc import simulate as mc
from alg1 import simulate as alg1

size = 100
lT = 0.5
hT = 10
h = 0
J = 1
# action_rates = np.empty((size, 2))
# action_rates[:, 0] = 0.7
# action_rates[:, 1] = 0.1
#action_rates = np.array([[0.7, 0.1] for i in range(size)])

action_rates = np.array([
    2 * [0.7 if i % 2 == 0 else 0.1]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))


def case_lt_without_correlations():
    """Produce decorrelated samples of MC and alg1 simulation at `T = 0.5`"""
    mc(
        size=size, temperature=lT, field=h, coupling=J,
        burn_in=1000, length=1000000, frame_step=100,
        output=f'{dir_path}/mc-lT.json'  # low T mc
    )
    alg1(
        size=size, temperature=lT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=1000, length=1000000, frame_step=100,
        output=f'{dir_path}/a1-lT.json'  # low T algorithm 1
    )


def case_ht_with_correlations():
    """Produce correlated samples of MC and alg1 simulation at `T = 0.5`"""
    mc(
        size=size, temperature=hT, field=h, coupling=J,
        burn_in=10, length=10000, frame_step=1,
        output=f'{dir_path}/mc-hT.json'  # low T mc
    )
    alg1(
        size=size, temperature=hT, field=h, coupling=J,
        action_rates=action_rates,
        burn_in=10, length=10000, frame_step=1,
        output=f'{dir_path}/a1-hT.json'  # low T algorithm 1
    )


if __name__ == '__main__':
    case_lt_without_correlations()
    case_ht_with_correlations()
