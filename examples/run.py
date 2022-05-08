"""Testing MC and A1 with three spins"""
# Add code directory to the python path
import sys, os
import numpy as np
import datetime

sys.path.append('code')
from mc import main as mc
from alg1 import main as alg1

code_path = os.path.dirname(os.path.realpath(__file__))


class Arguments:
    def __init__(self, entries):
        for k, v in zip(entries.keys(), entries.values()):
            self.__dict__.update(entries)


def case():
    """Produce samples of MC simulation at `T = 0.5` and `T = 10`
    """
    seed = 0
    size = 100
    lT = 0.5
    hT = 10
    h = 0
    J = 1
    action_rates = np.empty((size, 2))
    action_rates[:, 0] = 0.7
    action_rates[:, 1] = 0.1
    length = 100000
    frame_step = 10

    file_path = os.path.join(code_path, str(size) + '_spins')
    date_string = datetime.date.today().strftime("%Y-%m-%d")

    mc(Arguments({
        'size': size, 'temperature': lT, 'field': h, 'coupling': J, 'burn_in': 10, 'length': length,
        'seed': seed, 'frame_step': frame_step,
        'output': f'{file_path}/mc-lT_' + date_string + '.json'  # low T mc
    }))
    mc(Arguments({
        'size': size, 'temperature': hT, 'field': h, 'coupling': J, 'burn_in': 10, 'length': length,
        'seed': seed, 'frame_step': frame_step,
        'output': f'{file_path}/mc-hT_' + date_string + '.json'  # high T mc
    }))
    alg1(Arguments({
        'size': size, 'temperature': lT, 'field': h, 'coupling': J, 'action_rates': action_rates,
        'length': length, 'seed': seed, 'frame_step': frame_step,
        'output': f'{file_path}/alg1-lT_' + date_string + '.json' # low T alg1
    }))
    alg1(Arguments({
        'size': size, 'temperature': hT, 'field': h, 'coupling':J, 'action_rates': action_rates,
        'length': length, 'seed': seed, 'frame_step': frame_step,
        'output': f'{file_path}/alg1-hT_' + date_string + '.json'  # high T alg1
    }))


if __name__ == '__main__':
    case()
