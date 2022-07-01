"""Monte Carlo simulation of a one-dimensional Ising model

This script performs a Metropolis sampling of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse
import numpy as np
from tqdm import tqdm
import ising


def main(args):
    """@deprecated Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """
    energy = np.empty(len(args.temperature))
    magnetization = np.empty(len(args.temperature))
    for temp in range(len(args.temperature)):
        chain = ising.Metropolis(size=args.size, temperature=args.temperature[temp], field=args.field,
                                 coupling=args.coupling, seed=args.seed)
        np.random.seed(args.seed)

    # Skip the first burn_in samples so that the stationary distribution is reached
        for _ in tqdm(range(args.burn_in), desc="MC: Burn-in"): chain.advance()

        energy[temp] = chain.energy()
        magnetization[temp] = np.mean(chain.spins)

    return energy, magnetization


def simulate(
        size=3, temperature=np.array([0.5, 3]), field=0.0, coupling=1.0, burn_in=10,
        length=1000, seed=0, frame_step=1
):

    """Perform Metropolis sampling of an Ising chain

    The sample is output into a JSON file.

    Args:
        size (int): Chain size
        temperature (size): Heat bath temperature
        field (float): External magnetic field
        coupling (float): Spin coupling constant
        burn_in (int): Number of burn_in passes
        seed (int): Random generator seed
    """
    energy = np.empty(len(temperature))
    magnetization = np.empty(len(temperature))
    for temp in range(len(temperature)):
        chain = ising.Metropolis(size=size, temperature=temperature[temp], field=field, coupling=coupling, seed=seed)
        np.random.seed(seed)

        # Skip the first burn_in samples so that the stationary distribution is reached
        for _ in tqdm(range(burn_in), desc="MC: Burn-in"): chain.advance()

        energy[temp] = chain.energy()
        magnetization[temp] = np.mean(chain.spins)

    return energy, magnetization


### Execute main script


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-N', dest='size', type=int, default=3, help="Chain size"
    )
    parser.add_argument(
        '-T', dest='temperature', type=float, default=[0.5, 3.0],
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
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )

    main(parser.parse_args())
