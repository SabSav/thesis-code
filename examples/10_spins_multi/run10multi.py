"""Testing MC and A1 with three spins"""
import argparse, json
import sys, os
from functools import partial

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
length = 10000
frame_step = 10
dt_lT = 0.1
dt_hT = 0.04
temperatures = [0.5, 0.1]
dts = [0.1, 0.04]


def mc_seed(seed):

    return mc(size=size, temperature=temp, field=h, coupling=J, burn_in=burn_in, length=length, seed=seed,
              frame_step=frame_step)


def alg1_seed(seed):

    return alg1(size=size, temperature=temp, field=h, coupling=J, burn_in=burn_in, length=length, seed=seed,
                dt=frame_step, frame_step=frame_step, action_rates=action_rates)


def alg2_seed(seed):

    return alg2(size=size, temperature=temp, field=h, coupling=J, burn_in=burn_in, length=length, seed=seed,
                dt=frame_step, frame_step=frame_step, action_rates=action_rates)


def avg_quantities(simulation):
    a = np.array([x for x in zip(*simulation)])
    engy_mean = np.sum(a[0], axis=0) / len(simulation)
    m_mean = np.sum(a[1], axis=0) / len(simulation)
    return engy_mean, m_mean


def write_file(engy, m, filename):
    dict = {"coupling": J, "temperature": temp, "field": h, "number of spins": size, "energy_sample": engy.tolist(),
            "magnetization_sample": m.tolist(), "length": length, "frame_step": frame_step}
    with open(filename, 'w') as file:
        json.dump(dict, file)
    print(f"Simulations saved to {filename}")


for temp in temperatures:
    pos = temperatures.index(temp)
    dt = dts[pos]
    theory_cases = [
        {'method': theory, 'label': f'Temperature {temp} Theory',
         'size': size, 'temperature': temp, 'field': h, 'coupling': J,
         'output': f'{dir_path}/theory-T={temp}.json'},
    ]

    if __name__ == '__main__':

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

        with(Pool(processes=20)) as pool:
            sim_mc = np.array(pool.map(mc_seed, range(20)))
            sim_alg1 = pool.map(alg1_seed, range(20))
            sim_alg2 = pool.map(alg2_seed, range(20))

        mc_engy_avg, mc_m_avg = avg_quantities(sim_mc)
        alg1_engy_avg, alg1_m_avg = avg_quantities(sim_alg1)
        alg2_engy_avg, alg2_m_avg = avg_quantities(sim_alg2)

        write_file(mc_engy_avg, mc_m_avg, f'{dir_path}/mc-T={temp}.json')
        write_file(alg1_engy_avg, alg1_m_avg, f'{dir_path}/alg1-T={temp}.json')
        write_file(alg2_engy_avg, alg2_m_avg, f'{dir_path}/alg2-T={temp}.json')

