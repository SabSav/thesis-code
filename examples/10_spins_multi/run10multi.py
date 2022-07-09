"""Testing MC and A1 with three spins"""
import argparse, json
import sys, os
import numpy as np
import tqdm
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
burn_in_mc = np.array([1000000, 1400000])
burn_in_alg1 = np.array([1200000, 1200000])
burn_in_alg2 = np.array([1000000, 1000000])
dt_alg2 = np.array([2.0, 5.0])
dts = np.array([0.1, 0.04])
lT = 0.5
hT = 2
temperatures = np.array([lT, hT])


def mc_seed(seed):
    return mc(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in_mc, seed=seed)


def alg1_seed(seed):
    return alg1(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in_alg1, seed=seed,
                dt=dts, action_rates=action_rates)


def alg2_seed(seed):
    return alg2(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in_alg2, seed=seed,
                dt=dt_alg2, action_rates=action_rates)


def merge(simulation, alg, burn_in):
    filename = f'{dir_path}/{alg}'
    engy, m = np.array([x for x in zip(*simulation)])
    for temp in range(len(temperatures)):
        if temperatures[temp] == lT:
            dict = {"coupling": J, "temperature": lT, "field": h, "number of spins": size,
                    "energy_sample": engy[:, temp].tolist(), "magnetization_sample": m[:, temp].tolist(),
                    "burn-in": int(burn_in[temp])}
            filenamelT = f'{filename}-lT.json'
            with open(filenamelT, 'w') as file:
                json.dump(dict, file)
        elif temperatures[temp] == hT:
            dict = {"coupling": J, "temperature": hT, "field": h, "number of spins": size,
                    "energy_sample": engy[:, temp].tolist(), "magnetization_sample": m[:, temp].tolist(),
                    "burn-in": int(burn_in[temp])}
            filenamehT = f'{filename}-hT.json'
            with open(filenamehT, 'w') as file:
                json.dump(dict, file)
        else:
            print(f'Error: temperatures are not corresponding')

        print(f"Simulations saved to {filename}")


theory_cases = [
    {'method': theory, 'label': f'Temperature {lT} Theory',
     'size': size, 'temperature': lT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-lT.json'},
    {'method': theory, 'label': f'Temperature {hT} Theory',
     'size': size, 'temperature': hT, 'field': h, 'coupling': J,
     'output': f'{dir_path}/theory-hT.json'}
]

if __name__ == '__main__':

    with(Pool(processes=128)) as pool:
        sim_mc = pool.map(mc_seed, range(10000))
        sim_alg1 = pool.map(alg1_seed, range(10000))
        #sim_alg2 = pool.map(alg2_seed, range(10000))

    merge(sim_mc, 'mc', burn_in_mc)
    merge(sim_alg1, 'a1', burn_in_alg1)
    #merge(sim_alg2, 'a2', burn_in_alg2)

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
