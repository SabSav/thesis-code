"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse
import sys
from itertools import repeat
import numpy as np
import tqdm
import ising
from multiprocessing import Pool
import json


def main(args):
    """@deprecated Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """

    chain = ising.DynamicChain(size=args.size, temperature=args.temperature, coupling=args.coupling,
                               field=args.field, action_rates=args.action_rates, dt=args.dt)
    np.random.seed(args.seed)

    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm.tqdm(range(args.burn_in), desc="Alg1"): chain.advance()

    energy = chain.energy()
    magnetization = np.mean(chain.spins)

    return energy, magnetization


def simulate(
        seed, size, temperature, field, coupling, burn_in,
        action_rates, dt,
):
    chain = ising.DynamicChain(size=size, temperature=temperature, coupling=coupling,
                               field=field, action_rates=action_rates, dt=dt)
    np.random.seed(seed)

    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm.tqdm(range(burn_in), desc="Alg1"): chain.advance()

    energy = chain.energy()
    magnetization = np.mean(chain.spins)

    return energy, magnetization


def merge(output, size=None, temperature=None, field=None, coupling=None, burn_in=None,
          action_rates=None, dt=None):
    np.random.seed(0)
    seed = [np.random.randint(0, 2**32 - 1) for _ in range(10000)]
    with Pool(processes=128) as pool:
        simulation = pool.starmap(simulate, zip(seed, repeat(size), repeat(temperature), repeat(field),
                                                          repeat(coupling), repeat(burn_in), repeat(action_rates),
                                                          repeat(dt)))
    engy, m = np.array([x for x in zip(*simulation)])
    chain = ising.Metropolis(size=size, temperature=temperature, field=field, coupling=coupling)
    bundle = chain.export_dict()
    bundle["energy_sample"] = engy.tolist()
    bundle["magnetization_sample"] = m.tolist()
    bundle["burn-in"] = burn_in
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
        '-T', dest='temperature', type=float, default=0.5,
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
        '-AR', dest='action_rates', type=float, default=np.full((3, 2), 1),
        help="Action rates"
    )
    parser.add_argument(
        '-dt', dest='dt', type=float, default=np.array([0.1, 0.05]),
        help="Time interval"
    )

    parser.add_argument(
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )

    main(parser.parse_args())

