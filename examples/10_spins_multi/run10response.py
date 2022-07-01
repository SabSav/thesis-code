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
burn_in = 100
length = np.array([1000, 3000])
frame_step = np.array([1, 3])
temperatures = np.array([1, 1.1])
dt = [0.5, 0.04]


def mc_seed(seed):
    return mc(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
              length=length, frame_step=frame_step)


def alg1_seed(seed):
    return alg1(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=dt, action_rates=action_rates, length=length, frame_step=frame_step)


def alg2_seed(seed):
    return alg2(size=size, temperature=temperatures, field=h, coupling=J, burn_in=burn_in, seed=seed,
                dt=dt, action_rates=action_rates, length=length, frame_step=frame_step)


def avg_quantities(simulation, alg):
    simulation = np.mean(simulation, axis=0)
    dict = {"coupling": J, 'T - T0': temperatures[-1] - temperatures[0], "field": h, "number of spins": size,
            "Energy difference": simulation.tolist(), "burn-in": burn_in, "length temperature T0": int(length[0]),
            "frame_step temperature T0": int(frame_step[0]), "length temperature T": int(length[-1]),
            "frame_step temperature T": int(frame_step[-1])}
    """
    dict = {"coupling": J, "T - T0": temperatures[-1] - temperatures[0], "field": h, "number of spins": size,
            "Energy difference": simulation, "burn-in": burn_in, "length temperature T0": length[0],
            "frame_step temperature T0": frame_step[0], "length temperature T": length[-1],
            "frame_step temperature T": frame_step[-1]}
    """
    filename = f'{dir_path}/{alg}-response.json'
    with open(filename, 'w') as file:
        json.dump(dict, file)
    print(f"Simulations saved to {filename}")


if __name__ == '__main__':

    with(Pool(processes=20)) as pool:
        sim_mc = pool.map(mc_seed, range(20))
        sim_alg1 = pool.map(alg1_seed, range(20))
        sim_alg2 = pool.map(alg2_seed, range(20))

    avg_quantities(sim_mc, 'mc')
    avg_quantities(sim_alg1, 'alg1')
    avg_quantities(sim_alg2, 'alg2')

