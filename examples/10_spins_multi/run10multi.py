"""Testing MC and A1 with three spins"""
import argparse
import sys, os
from itertools import repeat

import numpy as np
# Add code directory to the python path
sys.path.append('code')
from mc_multi import write_file as mc
from alg1_multi import write_file as alg1
from alg2_multi import write_file as alg2
from theoretical_quantities import simulate as theory


dir_path = os.path.dirname(os.path.realpath(__file__))
size = 10
h = 0
J = 1
action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.3]
    for i in range(size)
])
lT = 0.5
hT = 2

cases = [
    {'method': theory, 'label': f'Temperature {lT} Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-lT.json'},
    {'method': theory, 'label': f'Temperature {hT} Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-hT.json'},
    {'method': mc, 'label': 'Low-temperature sample from an MC simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'burn_in': 2000, 'output': f'{dir_path}/mc-lT.json'},
    {'method': mc, 'label': 'High-temperature sample from an MC simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'burn_in': 100, 'output': f'{dir_path}/mc-hT.json'},
    {'method': alg1, 'label': 'Low-temperature sample from an A1 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 0.1, 'burn_in': 6000,
     'output': f'{dir_path}/a1-lT.json'},
    {'method': alg1, 'label': 'High-temperature sample from an A1 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1e-4, 'burn_in': 1500000,
     'output': f'{dir_path}/a1-hT.json'},
    {'method': alg2, 'label': 'Low-temperature sample from an A2 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1.0, 'burn_in': 1000,
     'output': f'{dir_path}/a2-lT.json'},
    {'method': alg2, 'label': 'High-temperature sample from an A1 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1.0, 'burn_in': 220,
     'output': f'{dir_path}/a2-hT.json'}

]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-i', dest='cases', type=str, default=':',
        help="Case indices (default: ':')"
    )
    args = parser.parse_args()

    for case in eval(f'np.array(cases)[{args.cases}]'):
        kwargs = case.copy()
        method = kwargs.pop('method')
        print(kwargs.pop('label'))
        method(**kwargs)
        print()
