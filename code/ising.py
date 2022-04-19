"""Simulation tools for Ising model"""
import numpy as np

class Chain:
    """One dimensional Ising model

    Attributes:
        spins (numpy.ndarray): Chain of up (+1) and down (-1) spin values
    """

    def __init__(self, coupling=1, temperature=1.0, field=0., **kwargs):
        """Initialize an Ising chain

        Args:
            coupling (int): Spin coupling-interaction constant is usually `1`
                (default) or `-1`.
            temperature (float): Temperature in Boltzmann-contant units
            field (float): External magnetic field
            size (int): Chain size must be greater than or equal to `3`
                (default). This keyword argument is ignored when the keyword
                argument `spins` is given.
            spins (array_like): When unspecified the chain of size `size` is
                initialized with randomly assigned spin values. Otherwise the
                chain size and the initial spin values are taken from `spins`.

        Raises:
            TypeError: When unknown argument is provided
        """
        self.coupling = coupling
        self.temperature = temperature
        self.field = field
        if 'spins' in kwargs:
            self.spins = np.asarray(kwargs['spins'])
            del kwargs['spins']
        else:
            size = 3
            if 'size' in kwargs:
                size = kwargs['size']
                del kwargs['size']
            self.spins = 2 * np.random.randint(2, size=size) - 1
        assert len(self.spins) >= 3

        if len(kwargs) > 0:
            raise TypeError(f'Unknown kwargs: {list(kwargs.keys())}')

    def deltaE(self, i):
        """Return energy cost of flipping a given spin"""
        return 2 * self.field * self.spins[i] + 2 * self.coupling * self.spins[i] * (
            self.spins[i - 1] + self.spins[(i + 1) % len(self.spins)]
        )
    def energy(self):
        """Return the chain energy"""
        size = len(self.spins)
        j_2 = self.coupling / 2
        engy = 0.0
        for i in range(size): engy -= self.spins[i] * (
            self.field + j_2 * (
                self.spins[i - 1] + self.spins[(i + 1) % size]
            )
        )
        return engy

    def export_dict(self):
        """Export dictionary containing system's parameters"""
        return {
            "coupling": self.coupling,
            "temperature": self.temperature, "field": self.field,
            "spins": self.spins.tolist()
        }

class DynamicChain(Chain):
    """An extension of the Ising chain for stochastic simulations

    Attributes:
        _buffer (np.ndarray): Buffer for simulation data
    """

    def __init__(self, coupling=1, temperature=1.0, field=0., **kwargs):
        """Overriden constructor with additional arguments

        Args:
            ...

            action_rates (array_like): An array of shape `(N, 2)` for `N` spin
                values `-1` and `1`
        """
        action_rates = (
            kwargs.pop('action_rates') if 'action_rates' in kwargs
            else None
        )
        super().__init__(coupling, temperature, field, **kwargs)

        self.action_rates = (
            np.full((len(self.spins), 2), 1.0) if action_rates is None
            else np.asarray(action_rates)
        )
        assert self.action_rates.shape == (len(self.spins), 2)

        self._buffer = np.empty_like(self.spins)

    def action_rate(self, i, value):
        """Return action rate for a given value of an `i`th spin"""
        return self.action_rates[i, (value + 1) // 2]

    def _prepare_buffer(self):
        """Return buffer after ensuring that it has the required size"""
        if self._buffer.shape != self.spins.shape:
            self._buffer = np.empty_like(self.spins)
        return self._buffer

    def advance(self):
        """Apply one simulation step of the algorithm I"""
        buffer = self._prepare_buffer()
        pass

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
