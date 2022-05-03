"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse, json
import numpy as np
from tqdm import tqdm
import ising

def main(args):
    """Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """

    chain = ising.Chain(temperature=args.temperature, size=args.size)
    np.random.seed(args.seed)
    energy = np.empty(args.length)
    magnetization = np.empty_like(energy)

    # Skip the first burn_in samples so that the stationary distribution is reached
    for _ in tqdm(range(args.burn_in), desc="Burn-in"):
        ising.metropolis_pass(chain)

    # Collect samples
    for i in tqdm(range(args.length), desc="Simulation"):
        ising.metropolis_pass(chain)
        energy[i] = chain.energy()
        magnetization[i] = np.mean(chain.spins)

    bundle = chain.export_dict()
    bundle['energy_sample'] = energy.tolist()
    bundle['magnetization_sample'] = magnetization.tolist()
    with open(args.output, 'w') as file: json.dump(bundle, file)
    print(f"Simulations saved to {args.output}")

    # import json
    # with open(path, 'r+') as file: bundle = json.load(file)
    # print("J is:", bundle["coupling"])


### Execute main script
if __name__== '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=str, help="Name of the output file")
    parser.add_argument(
        '-N', dest='size', type=int, default=3, help="Chain size"
    )
    parser.add_argument(
        '-T', dest='temperature', type=float, default=1.0,
        help="Temperature of the heat bath"
    )
    parser.add_argument(
        '-B', dest='burn_in', type=int, default=10,
        help="Number of burn-in passes"
    )
    parser.add_argument(
        '-l', dest='length', type=int, default=1000,
        help="Length of output samples"
    )
    parser.add_argument(
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )

    main(parser.parse_args())


def metropolis_pass(chain: Chain):
    """Apply Metropolis step to each spin of the chain in random order

    Returns
        Chain: Updated chain
    """

    # Iterate over spin indices taken in the random order
    for spin_to_change in np.random.permutation(np.arange(len(chain.spins))):
        dE = chain.deltaE(spin_to_change)

        # Metropolis condition min(1, exp(...)) always holds for dE < 0
        if np.random.random() < np.exp(- dE / chain.temperature):
            chain.spins[spin_to_change] *= -1

    return chain
