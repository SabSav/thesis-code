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
action_rates0 = np.array([
    2 * [0.05 if i % 2 == 0 else 0.08]
    for i in range(size)
])
action_rates1 = np.array([
    2 * [0.1 if i % 2 == 0 else 0.3]
    for i in range(size)
])
action_rates2 = np.array([
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
     'action_rates': action_rates0, 'dt': np.array([0.1, 0.1]),
     'burn_in': 1100, 'length': np.array([100, 500]), 'frame_step': np.array([10, 10]),
     'output': f'{dir_path}/a1-response-alphas={action_rates0[0,0]},{action_rates0[1,0]}.json'},
    {'method': alg2, 'label': 'Response Alg2',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates0, 'dt': np.array([1, 1]),
     'burn_in': 110, 'length': np.array([10, 50]), 'frame_step': np.array([1, 1]),
     'output': f'{dir_path}/a2-response-alphas={action_rates0[0,0]},{action_rates0[1,0]}.json'},
    {'method': alg1, 'label': 'Response Alg1',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates1, 'dt': np.array([0.1, 0.1]),
     'burn_in': 1100, 'length': np.array([100, 500]), 'frame_step': np.array([10, 10]),
     'output': f'{dir_path}/a1-response-alphas={action_rates1[0,0]},{action_rates1[1,0]}.json'},
    {'method': alg2, 'label': 'Response Alg2',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates1, 'dt': np.array([1, 1]),
     'burn_in': 110, 'length': np.array([10, 50]), 'frame_step': np.array([1, 1]),
     'output': f'{dir_path}/a2-response-alphas={action_rates1[0,0]},{action_rates1[1,0]}.json'},
    {'method': alg1, 'label': 'Response Alg1',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates2, 'dt': np.array([0.1, 0.1]),
     'burn_in': 1100, 'length': np.array([100, 500]), 'frame_step': np.array([10, 10]),
     'output': f'{dir_path}/a1-response-alphas={action_rates2[0,0]},{action_rates2[1,0]}.json'},
    {'method': alg2, 'label': 'Response Alg2',
     'size': size, 'temperature': np.array([lT, hT]), 'field': h, 'coupling': J,
     'action_rates': action_rates2, 'dt': np.array([1, 1]),
     'burn_in': 110, 'length': np.array([10, 50]), 'frame_step': np.array([1, 1]),
     'output': f'{dir_path}/a2-response-alphas={action_rates2[0,0]},{action_rates2[1,0]}.json'},
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



