"""Testing MC and A1 with three spins"""
import argparse, json
import sys, os

from tqdm.contrib import itertools
import numpy as np
# Add code directory to the python path
sys.path.append('code')
from mc_multi import simulate as mc
from alg1_multi import simulate as alg1
from theoretical_quantities import simulate as theory
from alg2_multi import simulate as alg2
from multiprocessing import Pool

size = 10
h = 0
J = 1
action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.05]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))
burn_in = 100
lT = 0.5
hT = 10
dts = np.array([0.5, 0.04])
temperatures = np.array([0.5, 10])


def mc_seed(seed):
    return mc(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed)


def alg1_seed(seed):
    return alg1(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=dts, action_rates=action_rates)


def alg2_seed(seed):
    return alg2(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=burn_in, action_rates=action_rates)


def merge(simulation, alg):
    engy, m = np.array([x for x in zip(*simulation)])
    for temp in range(len(temperatures)):
        dict = {"coupling": J, "temperatures": temperatures[temp], "field": h, "number of spins": size,
                "energy_sample": engy[:, temp].tolist(), "magnetization_sample": m[:, temp].tolist(),
                "burn-in": burn_in}
        filename = f'{dir_path}/{alg}-T={temperatures[temp]}.json'
        with open(filename, 'w') as file:
            json.dump(dict, file)
        print(f"Simulations saved to {filename}")


theory_cases = [
    {'method': theory, 'label': f'Temperature {lT} Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-T={lT}.json'},
    {'method': theory, 'label': f'Temperature {hT} Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-T={hT}.json'}
]

if __name__ == '__main__':

    with(Pool(processes=20)) as pool:
        sim_mc = pool.map(mc_seed, range(20))
        sim_alg1 = pool.map(alg1_seed, range(20))
        sim_alg2 = pool.map(alg2_seed, range(20))

    merge(sim_mc, 'mc')
    merge(sim_alg1, 'alg1')
    merge(sim_alg2, 'alg2')

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-i', dest='theory_cases', type=str, default=':',
        help="Case indices (default: ':')"
    )
    args = parser.parse_args()

    for case in eval(f'np.asarray(theory_cases)[{args.theory_cases}]'):
        kwargs = case.copy()
        method = kwargs.pop('method')
        print(kwargs.pop('label'))
        method(**kwargs)
        print()
