"""Testing MC and A1 with three spins"""
import argparse
import sys, os
import numpy as np

# Add code directory to the python path
sys.path.append('code')
from alg1 import simulate as alg1
from alg2 import simulate as alg2

size = 100
T = 2
h = 0
J = 1

action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.3]
    for i in range(size)
])

dt_a = 0.1
dt_b = 1e-3
dt_c = 1e-4

dir_path = os.path.dirname(os.path.realpath(__file__))

cases = [
    {'method': alg2, 'label': 'Low-temperature sample from an A1 simulation',
     'size': size, 'temperature': T, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1,
     'burn_in': 10, 'length': 10000, 'frame_step': 1,
     'output': f'{dir_path}/a2-hT.json',
     },
    {'method': alg1, 'label': 'Samples at temperature T with dts_a from an A1 simulation',
     'size': size, 'temperature': T, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_a,
     'burn_in': 10, 'length': 100000, 'frame_step': 10,
     'output': f'{dir_path}/a1-lT-{dt_a}.json',
     },
    {'method': alg1, 'label': 'Samples at temperature T with dts_b from an A1 simulation',
     'size': size, 'temperature': T, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_b,
     'burn_in': 10, 'length': 10**7, 'frame_step': 10**3,
     'output': f'{dir_path}/a1-hT-{dt_b}.json',
     },
    {'method': alg1, 'label': 'Samples at temperature T with dts_b from an A1 simulation',
     'size': size, 'temperature': T, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_b,
     'burn_in': 10, 'length': 10**8, 'frame_step': 10**4,
     'output': f'{dir_path}/a1-hT-{dt_b}.json',
     }

]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-i', dest='cases', type=str, default=':',
        help="Case indices (default: ':')"
    )
    args = parser.parse_args()

    for case in eval(f'np.asarray(cases)[{args.cases}]'):
        kwargs = case.copy()
        method = kwargs.pop('method')
        print(kwargs.pop('label'))
        method(**kwargs)
        print()

