"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse
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

    chain = ising.ContinuousDynamic(size=args.size, temperature=args.temperature, field=args.field,
                                    coupling=args.coupling, action_rates=args.action_rates, dt=args.dt, seed=args.seed)
    np.random.seed(args.seed)

    for spin in range(args.size): chain.random_times = chain.set_random_times(spin)
    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm.tqdm(range(args.burn_in), desc="MC"): chain.advance()

    energy = chain.energy()
    magnetization = np.mean(chain.spins)

    return energy, magnetization


def simulate(
        seed, size, temperature, field, coupling, action_rates, dt, burn_in
):
    chain = ising.ContinuousDynamic(size=size, temperature=temperature, field=field, coupling=coupling,
                                    action_rates=action_rates, dt=dt, seed=seed)
    np.random.seed(seed)

    # Skip the first burn_in samples so that the stationary distribution is reached
    for spin in range(size): chain.random_times = chain.set_random_times(spin)

    for _ in tqdm.tqdm(range(burn_in), desc="Alg2"): chain.advance()

    energy = chain.energy()
    magnetization = np.mean(chain.spins)
    return energy, magnetization, chain.spins, chain.random_times


def merge(size, temperature, field, coupling, action_rates, dt, burn_in):
    np.random.seed(0)
    seed = [np.random.randint(0, 2 ** 32 - 1) for _ in range(10000)]
    with Pool(processes=128) as pool:
        simulation = pool.starmap(simulate, zip(seed, repeat(size), repeat(temperature), repeat(field),
                                                repeat(coupling), repeat(action_rates), repeat(dt), repeat(burn_in)))
    engy, m, spins, random_times = [x for x in zip(*simulation)]

    return np.array(engy), np.array(m), np.array(spins), np.array(random_times)


def write_file(output, size, temperature, field, coupling, action_rates, dt,  burn_in):
    engy, m, spins, random_times = merge(size, temperature, field, coupling, action_rates, dt, burn_in)
    chain = ising.Chain(size=size, temperature=temperature, field=field, coupling=coupling)
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
