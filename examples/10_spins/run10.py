"""Testing MC and A1 with three spins"""
import argparse
import sys, os
import numpy as np

# Add code directory to the python path
sys.path.append('code')
from mc import simulate as mc
from alg1 import simulate as alg1
from theoretical_quantities import simulate as theory
from alg2 import simulate as alg2
size = 10
lT = 0.5
hT = 2
h = 0
J = 1
# action_rates = np.empty((size, 2))
# action_rates[:, 0] = 0.7
# action_rates[:, 1] = 0.1
# action_rates = np.array([[0.7, 0.1] for i in range(size)])

# action_rates = np.array([
#     2 * [0.7 if i % 2 == 0 else 0.1]
#     for i in range(size)
# ])
action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.05]
    for i in range(size)
])


dt_lT = 0.1
dt_hT = 0.02
dir_path = os.path.dirname(os.path.realpath(__file__))

cases = [
    {'method': theory, 'label': 'Low-temperature Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-lT.json'},
    {'method': theory, 'label': 'High-temperature Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-hT.json'},
    {'method': mc, 'label': 'Low-temperature sample from an MC simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'burn_in': 10000, 'length': 5000000, 'frame_step': 50,
     'output': f'{dir_path}/mc-lT.json'},
    {'method': mc, 'label': 'High-temperature sample from an MC simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'burn_in': 10000, 'length': 1000000, 'frame_step': 10,
     'output': f'{dir_path}/mc-hT.json'
     },
    {'method': alg1, 'label': 'Low-temperature sample from an A1 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_lT,
     'burn_in': 10000, 'length': 10000000, 'frame_step': 100,
     'output': f'{dir_path}/a1-lT.json'
     },
    {'method': alg1, 'label': 'High-temperature sample from an A1 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_hT,
     'burn_in': 10000, 'length': 50000000, 'frame_step': 500,
     'output': f'{dir_path}/a1-hT.json'
     },
    {'method': alg2, 'label': 'Low-temperature sample from an A2 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 300,
     'burn_in': 10000, 'length': 30000000, 'frame_step': 300,
     'output': f'{dir_path}/a2-lT.json'
     },
    {'method': alg2, 'label': 'High-temperature sample from an A2 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 250,
     'burn_in': 10000, 'length': 25000000, 'frame_step': 250,
     'output': f'{dir_path}/a2-hT.json'
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

