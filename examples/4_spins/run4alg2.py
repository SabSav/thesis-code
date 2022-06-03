"""Testing Alg2 with three spins"""
import sys, os
import numpy as np
import argparse

# Add code directory to the python path
sys.path.append('code')
from alg2 import simulate as alg2

size = 4
lT = 0.5
hT = 5
h = 0
J = -1

action_rates = np.array([
    2 * [0.7 if i % 2 == 0 else 0.1]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))

cases = [
    {'method': alg2, 'label': 'Low-temperature sample from an A2 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates,
     'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
     'output': f'{dir_path}/a2-lT.json'
     },
    {'method': alg2, 'label': 'High-temperature sample from an A2 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates,
     'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
     'output': f'{dir_path}/a2-hT.json'
     }
]

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
