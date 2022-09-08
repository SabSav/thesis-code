"""Testing MC and A1 with three spins"""
import argparse
import sys, os
import numpy as np
# Add code directory to the python path
sys.path.append('code')
from alg1_response import simulate as alg1
from alg2_response import simulate as alg2
from mc_response import simulate as mc

dir_path = os.path.dirname(os.path.realpath(__file__))
size = 100
h = 0
J = 1
action_rates = np.array([
    2 * [0.5 if i % 2 == 0 else 0.7]
    for i in range(size)
])
lT = 1.8
hT = 2.0

cases = [
    {'method': mc, 'label': 'Response MC',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'burn_in': 90, 'length': np.array([10, 50]), 'frame_step': np.array([1, 1]),
     'output': f'{dir_path}/mc-response.json'},
    {'method': alg1, 'label': 'Response Alg1',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': np.array([1e-3, 1e-3]),
     'burn_in': 170000, 'length': np.array([10**4, 5*10**4]), 'frame_step': np.array([10**3, 10**3]),
     'output': f'{dir_path}/a1-response-alphas={action_rates[0,0]},{action_rates[1,0]}.json'},
    {'method': alg2, 'label': 'Response Alg2',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': np.array([1, 1]),
     'burn_in': 162, 'length': np.array([10, 50]), 'frame_step': np.array([1, 1]),
     'output': f'{dir_path}/a2-response-alphas={action_rates[0,0]},{action_rates[1,0]}.json'}
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



