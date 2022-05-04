"""Testing MC and A1 with three spins"""
# Add code directory to the python path
import sys, os
sys.path.append('code')
from mc import main as mc

EXAMPLE_DIR = os.path.dirname(os.path.realpath(__file__))

class Arguments:
    def __init__(self, entries):
        for k, v in zip(entries.keys(), entries.values()):
            self.__dict__.update(entries)

def case_220504_mc():
    """Produce samples of MC simulation at two temperatures `T = 0.5`
    and `T = 10.0`
    """
    mc(Arguments({
        'size': 3, 'temperature': 0.5, 'burn_in': 10, 'length': 1000000,
        'seed': 0, 'frame_step': 100,
        'output': f'{EXAMPLE_DIR}/220504-mc-lt.json' # low temperature
    }))
    mc(Arguments({
        'size': 3, 'temperature': 10, 'burn_in': 10, 'length': 100000,
        'seed': 0, 'frame_step': 1,
        'output': f'{EXAMPLE_DIR}/220504-mc-ht.json' # high temperature
    }))
    mc(Arguments({
        'size': 3, 'temperature': 0.5, 'burn_in': 10, 'length': 1000000,
        'seed': 0, 'frame_step': 100,
        'output': f'{EXAMPLE_DIR}/220504-mc-lt-decorr.json' # decorrelated
    }))

### Execute main script
if __name__== '__main__':
    case_220504_mc()
