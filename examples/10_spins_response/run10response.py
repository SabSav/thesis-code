"""Testing MC and A1 with three spins"""
import json
import sys, os

import numpy as np

# Add code directory to the python path
sys.path.append('code')
from mc_response import simulate as mc
from alg1_response import simulate as alg1
from alg2_response import simulate as alg2
from multiprocessing import Pool

size = 10
h = 0
J = 1
action_rates = np.array([
    2 * [0.1 if i % 2 == 0 else 0.05]
    for i in range(size)
])

dir_path = os.path.dirname(os.path.realpath(__file__))
burn_in = 200000
length_mc = np.array([1000000, 5000000])
frame_step_mc = np.array([10, 50])
dt = 0.1
length_alg1 = np.array([10000000, 50000000])
frame_step_alg1 = np.array([100, 500])
length_alg2 = np.array([10000000, 50000000])
frame_step_alg2 = np.array([100, 500])
temperatures = np.array([1, 1.1])


def mc_seed(seed):
    return mc(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
              length=length_mc, frame_step=frame_step_mc)


def alg1_seed(seed):
    return alg1(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=dt, action_rates=action_rates, length=length_alg1, frame_step=frame_step_alg1)


def alg2_seed(seed):
    return alg2(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=frame_step_alg2, action_rates=action_rates, length=length_alg2, frame_step=frame_step_alg2)


def avg_quantities(simulation, alg, length, frame_step):
    simulation = np.mean(simulation, axis=0)
    dict = {"coupling": J, 'T0': temperatures[0], 'T':temperatures[-1], 'T - T0': temperatures[-1] - temperatures[0],
            'field': h, "number of spins": size, "Energy difference": simulation.tolist(), "burn-in": burn_in,
            "length temperature T0": int(length[0]), "frame_step temperature T0": int(frame_step[0]),
            "length temperature T": int(length[-1]), "frame_step temperature T": int(frame_step[-1])}
    filename = f'{dir_path}/{alg}-response.json'
    with open(filename, 'w') as file:
        json.dump(dict, file)
    print(f"Simulations saved to {filename}")


if __name__ == '__main__':

    with(Pool(processes=20)) as pool:
        sim_mc = pool.map(mc_seed, range(100))
        sim_alg2 = pool.map(alg2_seed, range(100))
        sim_alg1 = pool.map(alg1_seed, range(100))

    avg_quantities(sim_mc, 'mc', length_mc, frame_step_mc)
    avg_quantities(sim_alg1, 'alg1', length_alg1, frame_step_alg1)
    avg_quantities(sim_alg2, 'alg2', length_alg2, frame_step_alg2)

