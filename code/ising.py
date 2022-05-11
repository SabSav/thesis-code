"""Simulation tools for Ising model"""
import itertools
from collections import defaultdict
from collections import OrderedDict
import numpy
import numpy as np
import scipy
from scipy import special as sc
from scipy import stats
from tqdm import tqdm


class Chain:
    """One dimensional Ising model

        Attributes:
            spins (numpy.ndarray): Chain of up (+1) and down (-1) spin values
        """

    # 1) init is used to create an object
    # 2) kwards is used to pass a dictionary

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
        # self.something define a variable in the object

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
            random_seed = 1
            np.random.seed(random_seed)
            self.spins = 2 * np.random.randint(2, size=size) - 1
        assert len(self.spins) >= 3

        if len(kwargs) > 0:
            raise TypeError(f'Unknown kwargs: {list(kwargs.keys())}')

    def deltaE(self, i):
        """Return energy cost of flipping a given spin"""
        return 2 * self.spins[i] * (self.field + self.coupling * (self.spins[(i - 1) % len(self.spins)] +
                                                                  self.spins[(i + 1) % len(self.spins)]))

    def energy(self):
        """Return the chain energy"""
        size = len(self.spins)
        j_2 = self.coupling / 2
        engy = 0.0
        for i in range(size):
            engy -= self.spins[i] * (self.field + j_2 * (self.spins[(i - 1) % len(self.spins)] +
                                                         self.spins[(i + 1) % len(self.spins)]))
        return engy

    def export_dict(self):
        """Export dictionary containing system's parameters"""
        return {
            "coupling": self.coupling,
            "temperature": self.temperature, "field": self.field,
            "number of spins": len(self.spins)
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
            np.full((len(self.spins), 2), 1) if action_rates is None
            else np.asarray(action_rates)
        )  # define the variable action rate
        assert self.action_rates.shape == (len(self.spins), 2)

        self._buffer = np.empty_like(self.spins)

    def action_rate(self, i, value):
        """Return action rate for a given value of an `i`th spin (I need it for the step in Alg1)"""
        return self.action_rates[i, (value + 1) // 2]

    def _prepare_buffer(self):
        """Return buffer after ensuring that it has the required size"""
        if self._buffer.shape != self.spins.shape:
            self._buffer = np.empty_like(self.spins)
        return self._buffer

    def advance(self):
        """Apply one simulation step of the algorithm I"""
        buffer = self._prepare_buffer()
        buffer.fill(1)

        for spin_to_change in range(len(self.spins)):
            dE = self.deltaE(spin_to_change)

            # RB: That is lame hard coding =) You have `action_rates` keyword argument (kwarg),
            # RB: which you can use to specify you action rates. Do not hard-code your action rates!

            value = self.spins[spin_to_change]
            action_rates_ratio = self.action_rate(spin_to_change, -value) / self.action_rate(spin_to_change, value)
            weight = np.exp(-dE / self.temperature) * action_rates_ratio
            prob_change = self.action_rate(spin_to_change, value) * weight / (1 + weight)

            rank = np.random.random()
            if prob_change > rank:
                buffer[spin_to_change] = -1
        self.spins *= buffer
        return self.spins

class Metropolis(Chain):

    def __init__(self, coupling=1, temperature=1.0, field=0., **kwargs):
        super().__init__(coupling, temperature, field, **kwargs)

    def metropolis_pass(self):
        """Apply Metropolis step to each spin of the chain in random order

            Returns
                Chain: Updated chain
            """
        for spin_to_change in np.random.permutation(np.arange(len(self.spins))):
            if np.random.random() < np.exp(- self.deltaE(spin_to_change) / self.temperature):
                self.spins[spin_to_change] *= -1
        return self.spins


def acrl(m, time_evolution):
    m = np.asarray(m)
    m_demean = m - np.mean(m)
    nrm = np.var(m_demean)
    tot = len(m_demean)
    return np.array([
        np.mean(m_demean[t:] * m_demean[:tot - t])
        for t in range(time_evolution)
    ]) / nrm

def theoretical_quantities(chain: Chain, n_samples):

    theory_engy = []
    theory_m = []
    for conf in tqdm(itertools.product([1, -1], repeat=len(chain.spins)), desc='Generating theoretical configurations'):
        chain.spins = conf
        theory_m.append(np.mean(chain.spins))
        theory_engy.append(chain.energy())

    # RB: this can be majorly simplified
    theory_engy = np.sort(theory_engy)
    weights_config = np.exp(-(1 / chain.temperature) * theory_engy)
    config_prob = weights_config / sum(weights_config)
    keys = [float(value) for value in theory_engy]
    energy_prob = defaultdict(int)  # probability of a precise value of energy
    for k, n in zip(keys, config_prob):
        energy_prob[k] += n
    binomial_average = np.empty(len(energy_prob))
    binomial_std = np.empty(len(energy_prob))
    if len(chain.spins) <= 4: #adding this to calculate energy for spins chain > 4
        for i in range(len(energy_prob)):
            binomial_average[i] = scipy.stats.binom.mean(n=n_samples, p=list(energy_prob.values())[i])
            binomial_std[i] = scipy.stats.binom.std(n=n_samples, p=list(energy_prob.values())[i])

    theory_engy_counts = [value * n_samples for value in list(energy_prob.values())]

    return theory_engy, theory_m, theory_engy_counts, binomial_average, binomial_std


def count_variables(var):
    var = np.sort(var)
    keys = [float(value) for value in var]
    var_count = defaultdict(int)
    for k in zip(keys):
        var_count[k] += 1
    var_count = {key[0]: value for key, value in var_count.items()}
    return var_count


def std_algorithms(counts, theory_avg, theory_engy, theory_std):

    energy_level = defaultdict(int)
    keys = [float(value) for value in theory_engy]
    for k in zip(keys):
        energy_level[k] = 0
    i = 0
    for k in energy_level.keys():
        energy_level[k] = theory_avg[i]
        i += 1
    energy_level = {key[0]: value for key, value in energy_level.items()}

    items = list(counts.items())
    if len(energy_level) > len(counts.values()):
        for key in list(energy_level.keys()):
            if key not in list(counts.keys()):
                items.insert(0, (key, 0))
    counts = OrderedDict(sorted(dict(items).items()))

    multiplicity = np.empty(len(energy_level))
    std = theory_std
    for i in range(len(theory_avg)):
        factor = abs(list(counts.values())[i] - theory_avg[i]) / theory_std[i]
        multiplicity[i] = round(factor)
        if factor > 1:
            std[i] = factor * theory_std[i]

    return multiplicity, std, counts


def gof(f_obs, f_exp):
    """Calculate goodness of fit

    Args:
        f_obs (int[]): One-dimensional array of counts in a sample
        f_exp (int[]): Expected level of counts

    Returns: p-value (the goodness of fit)
    """
    k = len(f_exp) - 1 # number of degrees of freedom
    t_statistics = 0
    for i in range(len(f_exp)):
        t_statistics += pow(f_obs[i] - f_exp[i], 2) / f_obs[i]

    return sc.gammainc(k / 2, t_statistics / 2)

def two_sample_chi2test(dict_a, dict_b, n_samples_a, n_samples_b):
    k1 = pow(n_samples_b / n_samples_a, 1 / 2)
    k2 = pow(n_samples_a / n_samples_b, 1 / 2)

    total_keys = np.unique(list(dict_a.keys()) + list(dict_b.keys()))
    items_a = list(dict_a.items())
    items_b = list(dict_b.items())
    for key in total_keys:
        if key not in list(dict_a.keys()):
            items_a.insert(0, (key, 0))
        if key not in list(dict_b.keys()):
            items_b.insert(0, (key, 0))

    dict_a = OrderedDict(sorted(dict(items_a).items()))
    dict_b = OrderedDict(sorted(dict(items_b).items()))

    df = len(list(dict_a.values())) - 1
    t_statistics = 0
    for bin in range(len(list(dict_a.values()))):
        if list(dict_a.values())[bin] == 0 and list(dict_b.values())[bin] == 0:
            df -= 1
        else:
            t_statistics += pow(k1 * list(dict_a.values())[bin] -
                               k2 * list(dict_b.values())[bin], 2) / (list(dict_a.values())[bin] +
                                                                      list(dict_b.values())[bin])
    return sc.gammainc(0.5*df, t_statistics*0.5)


def hist(chain: Chain, vector_values, engy_flag):
    if engy_flag:
        bin_centers = sorted([value - 2 * abs(chain.coupling) for value in vector_values])
        bin_centers = [value + 4 * abs(chain.coupling) for value in bin_centers]
        bin_centers = [value - min(bin_centers) for value in bin_centers]
        bin_bound = np.append(bin_centers, bin_centers[-1] + 4 * chain.coupling)
        vector_values = [value - min(vector_values) for value in vector_values]
    else:
        bin_centers = sorted([value - 1 / len(chain.spins) for value in vector_values])
        bin_centers = [value + 2 / len(chain.spins) for value in bin_centers]
        bin_bound = np.append(bin_centers, bin_centers[-1] + 2 / len(chain.spins))

    return bin_bound, vector_values


if __name__ == '__main__':

    random_seed = 1
    np.random.seed(random_seed)
