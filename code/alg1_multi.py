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

    chain = ising.DynamicChain(size=args.size, temperature=args.temperature, coupling=args.coupling,
                               field=args.field, action_rates=args.action_rates, dt=args.dt)
    np.random.seed(args.seed)
    energy = np.empty(args.length // args.frame_step)
    magnetization = np.empty_like(energy)

    for _ in tqdm(range(args.burn_in), desc="Burn-in"): chain.advance()
    # Collect samples
    for i in tqdm(range(args.length), desc="Simulation"):
        chain.advance()
        if i % args.frame_step == 0:
            index = i // args.frame_step
            energy[index] = chain.energy()
            magnetization[index] = np.mean(chain.spins)

    print(f"Simulations saved to {args.output}")


def simulate(
        size=3, temperature=1.0, field=0.0, coupling=1.0, burn_in=10,
        length=1000, seed=0, frame_step=1, action_rates=None, dt=None,
):
    """Perform sampling of an Ising chain simulated by the Algoirthm 1

    The sample is output into a JSON file.

    Args:
        output (str): Path to the output file, which will be created or, if
            exists, overwritten
        size (int): Chain size
        temperature (float): Heat bath temperature
        field (float): External magnetic field
        coupling (float): Spin coupling constant
        burn_in (int): Number of burn_in passes
        length (int): Total length of the simulation
        seed (int): Random generator seed
        frame_step (int): Number of steps between the sampled frames
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

    chain = ising.DynamicChain(
        size=size, temperature=temperature, coupling=coupling,
        field=field, action_rates=action_rates, dt=dt
    )
    np.random.seed(seed)
    energy = np.empty(length // frame_step)
    magnetization = np.empty_like(energy)

    # Relax initial configuration
    for _ in tqdm(range(burn_in), desc="Burn-in"): chain.advance()

    # Collect samples
    for i in tqdm(range(length), desc="Simulation Alg1"):
        chain.advance()
        if i % frame_step == 0:
            index = i // frame_step
            energy[index] = chain.energy()
            magnetization[index] = np.mean(chain.spins)

    return energy, magnetization

### Execute main script


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-N', dest='size', type=int, default=3, help="Chain size"
    )
    parser.add_argument(
        '-T', dest='temperature', type=float, default=1.0,
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
        '-dt', dest='dt', type=float, default=0.1,
        help="Time interval"
    )
    parser.add_argument(
        '-l', dest='length', type=int, default=1000,
        help="Length of output samples"
    )
    parser.add_argument(
        '-S', dest='seed', type=int, default=0,
        help="Seed for the random number generator"
    )
    parser.add_argument(
        '-f', dest='frame_step', type=int, default=1,
        help="Frame step as a number of time steps"
    )

    main(parser.parse_args())
