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

    chain = ising.DynamicChain(size=args.size, temperature=args.temperature[0], coupling=args.coupling,
                               field=args.field, action_rates=args.action_rates, dt=args.dt[0], seed=args.seed)
    np.random.seed(args.seed)
    energy = np.empty(args.length[0] // args.frame_step[0] + args.length[1] // args.frame_step[1])

    for _ in tqdm(range(args.burn_in), desc="Alg1: Burn-in"): chain.advance()
    # Collect samples
    for i in tqdm(range(args.length[0]), desc="Alg1: Simulation temperature T0"):
        chain.advance()
        if i < args.length[0]:  # Collect samples for T = T0
            if i % args.frame_step[0] == 0:
                index = i // args.frame_step[0]
                energy[index] = chain.energy()

    chain.temperature = args.temperature[1]
    chain.dt = args.dt[1]
    for t in tqdm(range(args.length[1]), desc="Alg1: Simulation temperature T"):
        chain.advance()
        if t % args.frame_step[1] == 0:
            indexT = t // args.frame_step[0]
            energy[indexT + index] = chain.energy()
    return energy


def simulate(
        size=3, temperature=np.array([1.0, 1.1]), field=0.0, coupling=1.0, burn_in=100,
        length=np.array([1000, 3000]), seed=0, frame_step=np.array([1, 3]), action_rates=None, dt=np.array([0.1, 0.05]),
):

    chain = ising.DynamicChain(size=size, temperature=temperature[0], coupling=coupling,
                               field=field, action_rates=action_rates, dt=dt[0], seed=seed)
    np.random.seed(seed)
    energy = np.empty(length[0] // frame_step[0] + length[1] // frame_step[1])

    for _ in tqdm(range(burn_in), desc="Alg1: Burn-in"): chain.advance()

    for i in tqdm(range(length[0]), desc="Alg1: Simulation temperature T0"):
        chain.advance()
        if i < length[0]:  # Collect samples for T = T0
            if i % frame_step[0] == 0:
                index = i // frame_step[0]
                energy[index] = chain.energy()

    chain.temperature = temperature[1]
    chain.dt = dt[1]
    for t in tqdm(range(length[1]), desc="Alg1: Simulation temperature T"):
        chain.advance()
        if t % frame_step[1] == 0:
            indexT = t // frame_step[1]
            energy[indexT + index] = chain.energy()

    return energy

### Execute main script


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser(description=__doc__)
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