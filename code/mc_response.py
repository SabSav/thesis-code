"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse
from itertools import repeat
from multiprocessing import Pool
import numpy as np
import tqdm
import json
import ising


def main(args):
    """@deprecated Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """

    chain = ising.Metropolis(size=args.size, temperature=args.temperature[0], field=args.field, coupling=args.coupling,
                             seed=args.seed)
    np.random.seed(args.seed)
    energy = np.empty(args.length[0] // args.frame_step[0] + args.length[1] // args.frame_step[1])
    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm.tqdm(range(args.burn_in), desc="Burn-in MC"): chain.advance()

    for i in tqdm.tqdm(range(args.length[0]), desc="Simulation T0 MC"):
        chain.advance()
        if i < args.length[0]:  # Collect samples for T = T0
            if i % args.frame_step[0] == 0:
                index = i // args.frame_step[0]
                energy[index] = chain.energy()

    chain.temperature = args.temperature[1]
    for t in tqdm.tqdm(range(args.length[1]), desc="Simulation T MC"):
        chain.advance()
        if t % args.frame_step[1] == 0:
            indexT = t // args.frame_step[1]
            energy[indexT + index + 1] = chain.energy()

    return energy


def simulate(
        seed, size, temperature, field, coupling, burn_in, length, frame_step
):
    chain = ising.Metropolis(size=size, temperature=temperature[0], field=field, coupling=coupling,
                             seed=seed)
    np.random.seed(seed)
    energy = np.empty(length[0] // frame_step[0] + length[1] // frame_step[1])
    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm.tqdm(range(burn_in), desc="Burn-in MC"): chain.advance()

    for i in tqdm.tqdm(range(length[0]), desc="Simulation T0 MC"):
        chain.advance()
        if i < length[0]:  # Collect samples for T = T0
            if i % frame_step[0] == 0:
                index = i // frame_step[0]
                energy[index] = chain.energy()

    chain.temperature = temperature[1]
    for t in tqdm.tqdm(range(length[1]), desc="Simulation T MC"):
        chain.advance()
        if t % frame_step[1] == 0:
            indexT = t // frame_step[1]
            energy[indexT + index + 1] = chain.energy()

    return energy


def avg_trajectory(output, size, temperature, field, coupling, burn_in, length, frame_step):
    np.random.seed(0)
    seed = [np.random.randint(0, 2 ** 32 - 1) for _ in range(1000)]
    with Pool(processes=128) as pool:
        simulation = np.array(
            pool.starmap(simulate, zip(seed, repeat(size), repeat(temperature), repeat(field),
                                                 repeat(coupling), repeat(burn_in), repeat(length),
                                                 repeat(frame_step))))
    engy = np.mean(simulation, axis=0)
    dict = {"coupling": coupling, 'T0': temperature[0], 'T': temperature[-1], 'field': field, "number of spins": size,
            "burn-in": burn_in, "length temperature T0": int(length[0]),
            "frame_step temperature T0": int(frame_step[0]), "length temperature T": int(length[-1]),
            "frame_step temperature T": int(frame_step[-1]), "Energy difference": engy.tolist()}
    with open(output, 'w') as file:
        json.dump(dict, file)
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
        '-f', dest='frame_step', type=int, default=np.array([1, 3]),
        help="Frame step as a number of time steps"
    )

    main(parser.parse_args())
