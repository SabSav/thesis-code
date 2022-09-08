"""Testing MC and A1 with three spins"""
import sys, os, argparse
import numpy as np

# Add code directory to the python path
sys.path.append('code')
from mc import simulate as mc
from alg1 import simulate as alg1
from alg2 import simulate as alg2
from theoretical_quantities import simulate as theory

size = 3
lT = 0.5
hT = 3
h = 0
J = 1
action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.3]
    for i in range(size)
])

dt_lT = 0.1
dt_hT = 1e-3

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
     'burn_in': 10, 'length': 10000, 'frame_step': 1,
     'output': f'{dir_path}/mc-lT.json'},
    {'method': mc, 'label': 'High-temperature sample from an MC simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'burn_in': 10, 'length': 10000, 'frame_step': 1,
     'output': f'{dir_path}/mc-hT.json'
     },
    {'method': alg1, 'label': 'Low-temperature sample from an A1 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_lT,
     'burn_in': 10, 'length': 100000, 'frame_step': 10,
     'output': f'{dir_path}/a1-lT.json'
     },
    {'method': alg1, 'label': 'High-temperature sample from an A1 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': dt_hT,
     'burn_in': 10, 'length': 10000000, 'frame_step': 1000,
     'output': f'{dir_path}/a1-hT.json'
     },
    {'method': alg2, 'label': 'Low-temperature sample from an A2 simulation',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1,
     'burn_in': 10, 'length': 10000, 'frame_step': 1,
     'output': f'{dir_path}/a2-lT.json'
     },
    {'method': alg2, 'label': 'High-temperature sample from an A2 simulation',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'action_rates': action_rates, 'dt': 1,
     'burn_in': 10, 'length': 10000, 'frame_step': 1,
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

    for case in eval(f'np.asarray(cases)[{args.cases}]'):
        kwargs = case.copy()
        method = kwargs.pop('method')
        print(kwargs.pop('label'))
        method(**kwargs)
        print()

