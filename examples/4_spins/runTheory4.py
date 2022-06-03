"""Evaluate theoretical quantities"""
import sys, os, argparse

# Add code directory to the python path
sys.path.append('code')
from theoretical_quantities import simulate as theory

size = 4
lT = 0.5
hT = 5
h = 0
J = -1

dir_path = os.path.dirname(os.path.realpath(__file__))

cases = [
    {'method': theory, 'label': 'Low-temperature Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-lT.json'},
    {'method': theory, 'label': 'High-temperature Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-hT.json'}
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