"""Theoretical quantity of a one-dimensional Ising model

This script performs the theoretical evaluation of energy and magnetization of an Ising chain into
a given file in the JSON format.

Usage: python code/theory_quantity.py -h
"""

import argparse, json
import numpy as np
from tqdm import tqdm
import itertools
import ising


def main(args):
    chain = ising.Chain(size=args.size, temperature=args.temperature, coupling=args.coupling, field=args.field)
    eng = {}  # Energy levels with their weights
    mgn = {}  # Magnetization values (not mean) with their wights
    for conf in tqdm(
            itertools.product([1, -1], repeat=args.size), total=2 ** args.size,
            desc='Generating theoretical configurations'
    ):

        chain.spins = conf
        e = chain.energy()
        weight = np.exp(-e / chain.temperature)
        if e in eng:  # accumulate energy weights
            eng[e] += weight
        else:
            eng[e] = weight

        m = np.sum(chain.spins)
        if m in mgn:  # accumulate magnetization weights
            mgn[m] += weight
        else:
            mgn[m] = weight

    energy = np.array(sorted(eng.items()))
    magnetization = np.array(sorted(mgn.items()))

    z = np.sum(energy[:, 1])
    energy[:, 1] /= z
    magnetization[:, 0] /= args.size
    magnetization[:, 1] /= z


def simulate(output, size=3, temperature=1.0, field=0.0, coupling=1.0):
    """Evaluate theoretical quantity of an Ising chain

    The result is output into a JSON file.

    Args:
        output (str): Path to the output file, which will be created or, if
            exists, overwritten
        size (int): Chain size
        temperature (float): Heat bath temperature
        field (float): External magnetic field
        coupling (float): Spin coupling constant
        """

    chain = ising.Chain(size=size, temperature=temperature, coupling=coupling, field=field)
    eng = {}  # Energy levels with their weights
    mgn = {}  # Magnetization values (not mean) with their wights
    for conf in tqdm(
            itertools.product([1, -1], repeat=size), total=2 ** size,
            desc='Generating theoretical configurations'
    ):

        chain.spins = conf

        e = chain.energy()
        weight = np.exp(-e / chain.temperature)
        if e in eng:  # accumulate energy weights
            eng[e] += weight
        else:
            eng[e] = weight

        m = np.sum(chain.spins)
        if m in mgn:  # accumulate magnetization weights
            mgn[m] += weight
        else:
            mgn[m] = weight

    energy = np.array(sorted(eng.items()))
    magnetization = np.array(sorted(mgn.items()))

    z = np.sum(energy[:, 1])
    energy[:, 1] /= z
    magnetization[:, 0] /= size
    magnetization[:, 1] /= z

    bundle = chain.export_dict()
    bundle['energy_level'] = energy[:, 0].tolist()
    bundle['magnetization_level'] = magnetization[:, 0].tolist()
    bundle['energy_probs'] = energy[:, 1].tolist()
    bundle['magnetization_probs'] = magnetization[:, 1].tolist()
    with open(output, 'w') as file:
        json.dump(bundle, file)
    print(f"Simulations saved to {output}")


### Execute main script

if __name__ == '__main__':
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

    main(parser.parse_args())
