"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse

import ising
from alg1_multi import merge as a1_multi
import numpy as np
import tqdm
import json


def main(args):
    """@deprecated Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """
    engy0, m0, spins = a1_multi(size=args.size, temperature=args.temperature[0], field=args.field,
                                coupling=args.coupling, action_rates=args.action_rates, dt=args.dt[0],
                                burn_in=args.burn_in)
    chain = ising.DynamicChain(size=args.size, temperature=args.temperature[0], field=args.field,
                               coupling=args.coupling, action_rates=args.action_rates, dt=args.dt[0])
    np.random.seed(0)
    engy_T0 = np.empty(shape=(len(engy0), args.length[0] // args.frame_step[0]))
    engy_T = np.empty(shape=(len(engy0), args.length[1] // args.frame_step[1]))
    for traj in range(len(engy0)):
        chain.temperature = args.temperature[0]
        chain.dt = args.dt[0]
        engy_T0[traj, 0] = engy0[traj]
        for i in tqdm.tqdm(range(1, args.length[0]), desc="Simulation T0 Alg1"):
            chain.spins = spins[traj]
            chain.advance()
            if i % args.frame_step[0] == 0:
                index = i // args.frame_step[0]
                engy_T0[traj, index] = chain.energy()
        chain.temperature = args.temperature[1]
        chain.dt = args.dt[1]
        for t in tqdm.tqdm(range(args.length[1]), desc="Simulation T Alg1"):
            chain.advance()
            if t % args.frame_step[1] == 0:
                index = t // args.frame_step[1]
                engy_T[traj, index] = chain.energy()

    engy = np.concatenate((engy_T0, engy_T), axis=1)
    std_engy = np.std(engy, axis=0)
    engy = np.mean(engy, axis=0)


    bundle = chain.export_dict()
    del bundle['temperature']
    bundle['T0'] = float(args.temperature[0])
    bundle['T'] = float(args.temperature[1])
    bundle['Length T0'] = int(args.length[0])
    bundle['Length T'] = int(args.length[1])
    bundle['frame_step T0'] = int(args.frame_step[0])
    bundle['frame_step T'] = int(args.frame_step[1])
    bundle['energy_sample'] = engy.tolist()
    bundle['std_engy'] = std_engy.tolist()
    with open(args.output, 'w') as file:
        json.dump(bundle, file)
    print(f"Simulations saved to {args.output}")


def simulate(
        output, size, temperature, field, coupling, action_rates, dt, burn_in, length, frame_step
):
    engy0, m0, spins = a1_multi(size=size, temperature=temperature[0], field=field, coupling=coupling,
                                action_rates=action_rates, dt=dt[0], burn_in=burn_in)
    chain = ising.DynamicChain(size=size, temperature=temperature[0], field=field, coupling=coupling,
                               action_rates=action_rates, dt=dt[0])
    engy_T0 = np.empty(shape=(len(engy0), length[0] // frame_step[0]))
    engy_T = np.empty(shape=(len(engy0), length[1] // frame_step[1]))
    for traj in range(len(engy0)):
        chain.temperature = temperature[0]
        chain.dt = dt[0]
        engy_T0[traj, 0] = engy0[traj]
        for i in tqdm.tqdm(range(1, length[0]), desc="Simulation T0 Alg1"):
            chain.spins = spins[traj]
            chain.advance()
            if i % frame_step[0] == 0:
                index = i // frame_step[0]
                engy_T0[traj, index] = chain.energy()
        chain.temperature = temperature[1]
        chain.dt = dt[1]
        for t in tqdm.tqdm(range(length[1]), desc="Simulation T Alg1"):
            chain.advance()
            if t % frame_step[1] == 0:
                index = t // frame_step[1]
                engy_T[traj, index] = chain.energy()

    engy = np.concatenate((engy_T0, engy_T), axis=1)
    std_engy = np.std(engy, axis=0)
    engy = np.mean(engy, axis=0)


    bundle = chain.export_dict()
    del bundle['temperature']
    bundle['T0'] = float(temperature[0])
    bundle['T'] = float(temperature[1])
    bundle['Length T0'] = int(length[0])
    bundle['Length T'] = int(length[1])
    bundle['frame_step T0'] = int(frame_step[0])
    bundle['frame_step T'] = int(frame_step[1])
    bundle['energy_sample'] = engy.tolist()
    bundle['std_engy'] = std_engy.tolist()
    with open(output, 'w') as file:
        json.dump(bundle, file)
    print(f"Simulations saved to {output}")


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=str, help="Name of the output file")
    parser.add_argument(
        '-N', dest='size', type=int, default=3, help="Chain size"
    )
    parser.add_argument(
        '-T', dest='temperature', type=float, default=np.array([1.0, 1.1]),
        help="Temperature of the heat bath"
    )
    parser.add_argument(
        '-h', dest='field', type=float, default=0.0, help="External field"
    )
    parser.add_argument(
        '-J', dest='coupling', type=float, default=1.0,
        help="Interaction term"
    )
    parser.add_argument(
        '-AR', dest='action_rates', type=float, default=np.full((3, 2), 1),
        help="Action rates"
    )
    parser.add_argument(
        '-dt', dest='dt', type=float, default=np.array([0.1, 0.05]),
        help="Time interval"
    )
    parser.add_argument(
        '-B', dest='burn_in', type=int, default=10,
        help="Number of burn-in passes"
    )
    parser.add_argument(
        '-l', dest='length', type=int, default=np.array([1000, 3000]),
        help="Length of output samples"
    )
    parser.add_argument(
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )
    parser.add_argument(
        '-f', dest='frame_step', type=int, default=np.array([10, 10]),
        help="Frame step as a number of time steps"
    )

    main(parser.parse_args())
