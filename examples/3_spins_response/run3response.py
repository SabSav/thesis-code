"""Testing MC and A1 with three spins"""
import argparse
import sys, os
import numpy as np
# Add code directory to the python path
sys.path.append('code')
from mc_response import simulate as mc
from alg1_response import simulate as alg1
from alg2_response import simulate as alg2
from theoretical_quantities import simulate as theory

dir_path = os.path.dirname(os.path.realpath(__file__))
size = 3
h = 0
J = 1
action_rates = np.array([
    2 * [0.2 if i % 2 == 0 else 0.1]
    for i in range(size)
])
lT = 0.5
hT = 1

cases = [
    {'method': theory, 'label': f'Temperature {lT} Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-lT.json'},
    {'method': theory, 'label': f'Temperature {hT} Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-hT.json'},
    {'method': mc, 'label': 'Response MC',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'burn_in': 5500, 'length': np.array([500, 1000]), 'frame_step':np.array([10, 10]),
     'output': f'{dir_path}/mc-response.json'},
    {'method': alg1, 'label': 'Response Alg1',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J, 'action_rates': action_rates,
     'dt':np.array([0.1, 0.1]), 'burn_in': 50000, 'length': np.array([500, 1000]),
     'frame_step': np.array([10, 10]), 'output': f'{dir_path}/a1-response.json'},
    {'method': alg2, 'label': 'Response Alg1',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J, 'action_rates': action_rates,
     'dt':np.array([10, 10]), 'burn_in': 8000, 'length': np.array([500, 1000]),
     'frame_step': np.array([10, 10]), 'output': f'{dir_path}/a2-response.json'}

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



