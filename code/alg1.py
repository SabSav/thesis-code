"""Algorithm 1 simulation of a one-dimensional Ising model

This script performs the dynamic simulation of an Ising chain and outputs parameters
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

    chain = ising.DynamicChain(size=args.size, temperature=args.temperature, coupling=args.coupling,
                               field=args.field, action_rates=args.action_rates)
    np.random.seed(args.seed)
    energy = np.empty(args.length // args.frame_step)
    magnetization = np.empty_like(energy)

    # Collect samples
    for i in tqdm(range(args.length), desc="Simulation"):
        chain.advance(index_criterion=True)
        if i % args.frame_step == 0:
            index = i // args.frame_step
            energy[index] = chain.energy()
            magnetization[index] = np.mean(chain.spins)

    bundle = chain.export_dict()
    bundle['energy_sample'] = energy.tolist()
    bundle['magnetization_sample'] = magnetization.tolist()
    with open(args.output, 'w') as file:
        json.dump(bundle, file)
    print(f"Simulations saved to {args.output}")


### Execute main script

if __name__ == '__main__':
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
