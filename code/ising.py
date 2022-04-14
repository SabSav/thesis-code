"""Simulation of Ising model"""

import numpy as np

class Chain:
    """One dimensional Ising model

    Attributes:
        spins (numpy.ndarray): Chain of up (+1) and down (-1) spin values
    """

    def __init__(self, coupling = 1, field = 0., **kwargs):
        """Initialize an Ising chain

        Args:
            coupling (int): Spin coupling-interaction constant is usually 1
                (default) or -1.
            field (float): External magnetic field
            size (int): Chain size must be greater than or equal to 3 (default).
                This keyword argument is ignored when the keyword argument
                `spins` is given.
            spins (None | array_like): When `None` the chain of size `size` is
                initialized with randomly assigned spin values. Otherwise the
                chain size and the initial spin values are taken from `spins`.
        """
        self.coupling = coupling
        self.field = 0
        if 'spins' in kwargs:
            self.spins = np.asarray(kwargs['spins'])
        else:
            size = kwargs['size'] if 'size' in kwargs else 3
            self.spins = 2 * np.random.randint(2, size=size) - 1
        assert len(self.spins) >= 3

    def deltaE(self, i):
        """Return energy cost of flipping a given spin"""
        return 2 * self.field * self.spins[i] + self.coupling * self.spins[i] * (
            self.spins[i - 1] + self.spins[(i + 1) % len(self.spins)]
        )
