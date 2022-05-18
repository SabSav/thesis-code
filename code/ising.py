"""Simulation tools for Ising model"""
import itertools
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import numpy
import numpy as np
import scipy
from scipy import stats
from tqdm import tqdm


class Chain:
    """One dimensional Ising model

        Attributes:
            spins (numpy.ndarray): Chain of up (+1) and down (-1) spin values
        """

    # 1) init is used to create an object
    # 2) kwards is used to pass a dictionary

    def __init__(self, coupling=1.0, temperature=1.0, field=0., **kwargs):
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

    def __init__(self, coupling=1.0, temperature=1.0, field=0., **kwargs):
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


def acrl(v, time_evolution):
    """Calculate the correlation for both quantity v
    Args:
        v: 1D array of samples
        time_evolution:
    Returns: correlation
    """
    v = np.asarray(v)
    v_demean = v - np.mean(v)
    nrm = np.var(v_demean)
    tot = len(v_demean)

    return np.array([
        np.mean(v_demean[t:] * v_demean[:tot - t])
        for t in range(time_evolution)
    ]) / nrm


def theoretical_distributions(chain: Chain):
    """Return theoretical distributions for the energy and magnetization

    The estimates are returned within a reasonable time with 25 spins or less.

    Args:
        chain: Ising chain for which the estimates are calculated

    Returns:
        (float, (n, 2)): `n` energy levels with values in the first column
            and their probabilities in the second (sorted by the energy)

        (float, (n, 2)): `n` magnetization states with values in the first
            column and their probabilities in the second (sorted by values)
    """

    size = len(chain.spins)
    eng = {}  # Energy levels with their weights
    mgn = {}  # Magnetization values (not mean) with their wights
    for conf in tqdm(
            itertools.product([1, -1], repeat=size), total=2 ** size,
            desc='Generating theoretical configurations'
    ):

        chain.spins = conf

        e = chain.energy()
        weight = np.exp(-e / chain.temperature)
        if e in eng: # accumulate energy weights
            eng[e] += weight
        else:
            eng[e] = weight

        m = np.sum(chain.spins)
        if m in mgn: # accumulate magnetization weights
            mgn[m] += weight
        else:
            mgn[m] = weight

    energy = np.array(sorted(eng.items()))
    magnetization = np.array(sorted(mgn.items()))

    z = np.sum(energy[:, 1])
    energy[:, 1] /= z
    magnetization[:, 0] /= size
    magnetization[:, 1] /= z

    return energy, magnetization


def theoretical_quantities(n_samples, quantity):
    """Calculate the theoretical energy and magnetization counts.
       Calculate the binomial avg and std deviation for theoretical energy

        Args:
            n_samples: total number of samples of the algorithms
            quantity: theoretical quantity

        Returns:
            (float, 1D-array): counts per theoretical quantity level
            (float): multinomial average
            (float): multinomial standard deviation
        """
    theory_quantity_counts = [quantity[i, 1] * n_samples for i in range(len(quantity))]

    binom_average = np.empty(len(quantity))
    binom_std = np.empty(len(quantity))

    for i in range(len(quantity)):
        binom_average[i] = scipy.stats.binom.mean(n_samples, p=quantity[i, 1])
        binom_std[i] = scipy.stats.binom.std(n=n_samples, p=quantity[i, 1])

    return theory_quantity_counts, binom_average, binom_std


def std_algorithms(counts, theory_avg, theory_quantity_counts, theory_std):
    """Calculate the std for the two algorithms for the barchart (used when N.spins <= 4)
        Args:
            counts: counts per quantity level generating by the algorithm
            theory_avg: binomial avg
            theory_engy: theoritical counts per quantity level
            theory_std: binomial std

        Returns: multiplicity: integer value that multiply the theory_std
                 std: actual std of the algorithms for each bar
                 counts: dictionary with the counts per energy level
    """

    quantity_level = defaultdict(int)
    keys = theory_quantity_counts[:, 0]

    for k in zip(keys):
        quantity_level[k] = 0
    i = 0
    for k in quantity_level.keys():
        quantity_level[k] = theory_avg[i]
        i += 1
    quantity_level = {key[0]: value for key, value in quantity_level.items()}

    items = list(counts.items())
    if len(quantity_level) > len(counts.values()):
        for key in list(quantity_level.keys()):
            if key not in list(counts.keys()):
                items.insert(0, (key, 0))
    counts = OrderedDict(sorted(dict(items).items()))

    multiplicity = np.empty(len(quantity_level))
    std = theory_std
    for i in range(len(theory_avg)):
        factor = abs(list(counts.values())[i] - theory_avg[i]) / theory_std[i]
        multiplicity[i] = round(factor)
        if factor > 1:
            std[i] = factor * theory_std[i]

    return multiplicity, std, counts


def count_variables(var):
    """Calculate the counts for energy level/magnetization values
        Args:
            var: samples of the algorithm

        Returns: dictionary with the counts per energy levels/ magnetization values
    """
    var = np.sort(var)
    var_count = Counter(var)
    return var_count


def test_theoretical(f_obs, theory_prob, n_samples, eps):
    """Testing for equivalence of a single multinomial distribution with a fully specified reference distribution
       upper bound = eps^2 - qnorm(1-alpha)*std
       distance = (prob_sample - theory_prob)**2
       Reject lack of fit if distance < upper_bound
            Args:
                f_obs: counts of the quantity to analyze
                theory_prob: theoretical probabilities
                n_samples: total number of samples
                eps: tolerance level

            Returns: dictionary with:
                    - distance
                    - upper bound
                    - reject: if True the two distribution are the same, False the two dristibution are not the same
    """
    prob_sample = np.array([i/n_samples for i in f_obs])
    dist_sample = 0
    var1 = 0
    var2 = 0
    for i in range(len(theory_prob)):
        dist_sample += (prob_sample[i] - theory_prob[i])**2
        var1 += (prob_sample[i] - theory_prob[i])**2 * prob_sample[i]
    for i in range(len(theory_prob)):
        for j in range(len(theory_prob)):
            var2 += (prob_sample[i] - theory_prob[i]) * (prob_sample[j] - theory_prob[j]) * prob_sample[i] * prob_sample[j]
    std = np.sqrt((4 * (var1 - var2))/n_samples)
    alpha = 0.05
    upper_bound = eps**2 - scipy.stats.norm.ppf(1 - alpha)*std
    reject = False
    if dist_sample < upper_bound:
        reject = True
    results = {"Distance": dist_sample, "Upper_bound": upper_bound, "Reject": reject}
    return results


def z_test(f1, f2, n_samples):
    """Z-test for the counts of the two algorithms. It assumes independence in the samples (articles)
    :param f1: counts per quantity level of the first algorithm
    :param f2: counts per quantity level of the second algorithm
    :param n_samples:
    :return: p-value of z-test
    """
    prob_sample1 = np.array([i/n_samples for i in f1])
    prob_sample2 = np.array([i / n_samples for i in f2])
    diff_rv = 0
    for i in range(len(f1)):
        diff_rv += prob_sample1[i] - prob_sample2[i]
    std = np.sqrt(np.var(prob_sample1)/n_samples + np.var(prob_sample2)/n_samples)
    return 2*scipy.stats.norm.sf(x=abs(diff_rv), loc=0, scale=std)


def equilibrate_counts(dict_a, dict_b):
    total_keys = np.unique(list(dict_a.keys()) + list(dict_b.keys()))
    for key in total_keys:
        if key not in dict_a.keys():
            dict_a[key] = 0
        if key not in dict_b.keys():
            dict_b[key] = 0

    # RB: Just `sorted(dict_*.items())` -> non funziona
    dict_a = OrderedDict(sorted(dict_a.items()))
    dict_b = OrderedDict(sorted(dict_b.items()))

    return dict_a, dict_b


def hist(chain: Chain, vector_values, engy_flag):
    """Define the bin_bound of the histogram per energy and magnetization
        Args:
            chain: class used to access at the coupling/spins value
            vector_values: samples of the algorithm
            engy_flag: flag used because in the energy histogram we have to rescale to E - Emin the plot

        Returns: dictionary with the counts per energy levels/ magnetization values
    """
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
