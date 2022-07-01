"""Algorithm 1 simulation of a one-dimensional Ising model

This script performs the dynamic simulation of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/mc.py -h
"""
import argparse
import numpy as np
from tqdm import tqdm
import ising


def main(args):
    """Perform Metropolis sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """
    energy = np.empty(len(args.temperature))
    magnetization = np.empty(len(args.temperature))
    for temp in range(len(args.temperature)):
        chain = ising.DynamicChain(size=args.size, temperature=args.temperature[temp], coupling=args.coupling,
                                   field=args.field, action_rates=args.action_rates, dt=args.dt[temp])
        np.random.seed(args.seed)

        for _ in tqdm(range(args.burn_in), desc="Alg1: Burn-in"): chain.advance()

        energy[temp] = chain.energy()
        magnetization[temp] = np.mean(chain.spins)
        # Collect samples
    return energy, magnetization


def simulate(
        size=3, temperature=np.array([0.5, 3]), field=0.0, coupling=1.0, burn_in=10,
        seed=0, action_rates=None, dt=np.array([0.1, 0.05]),
):
    """Perform sampling of an Ising chain simulated by the Algoirthm 1

    The sample is output into a JSON file.

    Args:
        size (int): Chain size
        temperature (float): Heat bath temperature
        field (float): External magnetic field
        coupling (float): Spin coupling constant
        burn_in (int): Number of burn_in passes
        seed (int): Random generator seed
        action_rates (int[]): Action rates of the spins
        :param burn_in:
        :param coupling:
        :param field:
        :param temperature:
        :param output:
        :param size:
        :param length:
        :param seed:
        :param frame_step:
        :param action_rates:
        :param dt:
    """

    energy = np.empty(len(temperature))
    magnetization = np.empty(len(temperature))
    for temp in range(len(temperature)):
        chain = ising.DynamicChain(size=size, temperature=temperature[temp], coupling=coupling,
                                   field=field, action_rates=action_rates, dt=dt[temp])
        np.random.seed(seed)

        for _ in tqdm(range(burn_in), desc="Alg1: Burn-in"): chain.advance()

        energy[temp] = chain.energy()
        magnetization[temp] = np.mean(chain.spins)
        # Collect samples
    return energy, magnetization


### Execute main script


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-N', dest='size', type=int, default=3, help="Chain size"
    )
    parser.add_argument(
        '-T', dest='temperature', type=float, default=np.array([0.5, 3]),
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
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )

    main(parser.parse_args())
