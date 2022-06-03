"""Algorithm 2 simulation of a one-dimensional Ising model

This script performs the continuous dynamic simulation of an Ising chain and outputs parameters
of the simulation together with a sample of energy and average magnetization into
a given file in the JSON format.

Usage: python code/alg2.py -h
"""
import argparse, json
import numpy as np
from tqdm import tqdm
import ising


def main(args):
    """Perform Alg2 sampling of an Ising chain

    Args:
        args: Parsed command-line arguments
    """

    chain = ising.ContinuousDynamic(size=args.size, temperature=args.temperature, coupling=args.coupling,
                                    field=args.field, action_rates=args.action_rates, dt=args.dt,
                                    random_times=args.random_times)
    np.random.seed(args.seed)
    energy = np.empty(args.length // args.frame_step)
    magnetization = np.empty_like(energy)

    for spin in range(args.size): chain.random_times = chain.set_random_times(spin)

    for _ in tqdm(range(args.burn_in), desc="Burn-in"): chain.continuous_advance()

    for t in tqdm(range(args.length), desc="Simulation"):
        chain.continuous_advance()
        if t % args.frame_step == 0:
            index = t // args.frame_step
            energy[index] = chain.energy()
            magnetization[index] = np.mean(chain.spins)


def simulate(
        output, size=3, temperature=1.0, field=0.0, coupling=1.0, burn_in=10,
        length=1000, seed=0, frame_step=1, action_rates=None, dt=None,
):
    """Perform sampling of an Ising chain simulated by the Algoirthm 2

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
    """

    chain = ising.ContinuousDynamic(size=size, temperature=temperature, coupling=coupling, field=field,
                                    action_rates=action_rates, dt=dt)
    np.random.seed(seed)
    energy = np.empty(length // frame_step)
    magnetization = np.empty_like(energy)

    for spin in range(size): chain.random_times = chain.set_random_times(spin)

    for _ in tqdm(range(burn_in), desc="Burn-in"): chain.continuous_advance()

    for t in tqdm(range(length), desc="Simulation"):
        chain.continuous_advance()
        if t % frame_step == 0:
            index = t // frame_step
            energy[index] = chain.energy()
            magnetization[index] = np.mean(chain.spins)

    bundle = chain.export_dict()
    bundle['energy_sample'] = energy.tolist()
    bundle['magnetization_sample'] = magnetization.tolist()
    with open(output, 'w') as file:
        json.dump(bundle, file)
    print(f"Simulations saved to {output}")


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
