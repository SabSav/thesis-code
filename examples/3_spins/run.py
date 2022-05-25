"""Testing MC and A1 with three spins"""
import sys, os, argparse
import numpy as np
import datetime

# Add code directory to the python path
sys.path.append('code')
from mc import simulate as mc
from alg1 import simulate as alg1

size = 3
lT = 0.5
hT = 10
h = 0
J = 1
# action_rates = np.empty((size, 2))
# action_rates[:, 0] = 0.7
# action_rates[:, 1] = 0.1
#action_rates = np.array([[0.7, 0.1] for i in range(size)])

# action_rates = np.array([
#     2 * [0.7 if i % 2 == 0 else 0.1]
#     for i in range(size)
# ])
action_rates = np.array([
    2 * [0.7 if i % 2 == 0 else 0.1]
    for i in range(size)
])

dt_lT = 0.2
dt_hT = 0.1

dir_path = os.path.dirname(os.path.realpath(__file__))

cases = [
    {'method': mc, 'label': 'Low-temperature sample from an MC simulation',
        'size': size, 'temperature': lT, 'field': h, 'coupling': J,
        'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
        'output': f'{dir_path}/mc-lT.json'},
    {'method': mc, 'label': 'High-temperature sample from an MC simulation',
        'size': size, 'temperature': hT, 'field': h, 'coupling': J,
        'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
        'output': f'{dir_path}/mc-hT.json'
    },
    {'method': alg1, 'label': 'Low-temperature sample from an A1 simulation',
        'size': size, 'temperature': lT, 'field': h, 'coupling': J,
        'action_rates': action_rates, 'dt': dt_lT,
        'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
        'output': f'{dir_path}/a1-lT.json'
    },
    {'method': alg1, 'label': 'High-temperature sample from an A1 simulation',
        'size': size, 'temperature': hT, 'field': h, 'coupling': J,
        'action_rates': action_rates, 'dt': dt_hT,
        'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
        'output': f'{dir_path}/a1-hT.json'
    }
]


# def case_hT_with_correlations():
#     """Produce correlated samples of MC simulation at `T = 0.5`"""
#     mc(
#         size=size, temperature=lT, field=h, coupling=J,
#         burn_in=4000, length=100000, frame_step=1,
#         output=f'{dir_path}/mc-lT.json'  # low T mc
#     )
#     alg1(
#         size=size, temperature=lT, field=h, coupling=J,
#         action_rates=action_rates,
#         burn_in=100000, length=10000000, frame_step=100,
#         output=f'{dir_path}/a1-lT.json'  # low T algorithm 1
#     )
#     # alg1(
#     #     size=size, temperature=hT, field=h, coupling=J,
#     #     action_rates=action_rates,
#     #     burn_in=100000, length=10000000, frame_step=100,
#     #     output=f'{dir_path}/a1-hT.json'  # low T algorithm 1
#     # )

# def case_without_correlations():
#     """Produce decorrelated samples of MC simulation at `T = 0.5` and `T = 10`"""
#
#     mc(
#         size=size, temperature=lT, field=h, coupling=J,
#         burn_in=10, length=10000000, frame_step=10000,
#         output=f'{dir_path}/mc-lT-decor.json'  # low T mc
#     )
#     alg1(
#         size=size, temperature=lT, field=h, coupling=J,
#         action_rates=action_rates,
#         burn_in=10, length=10000000, frame_step=10000,
#         output=f'{dir_path}/a1-lT-decor.json'  # low T algorithm 1
#     )

if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', dest='cases', type=str, default=':',
        help="Case indices (default: ':')"
    )
    args = parser.parse_args()

    for case in eval(f'cases[{args.cases}]'):
        kwargs = case.copy()
        method = kwargs.pop('method')
        print(kwargs.pop('label'))
        method(**kwargs)
        print()
