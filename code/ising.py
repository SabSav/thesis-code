"""Simulation tools for Ising model"""
import itertools
from collections import defaultdict
from collections import OrderedDict
import numpy
import numpy as np
import scipy
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from copy import copy
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
            self.spins = np.asarray(kwargs['spins'])  # se diamo la catena iniziale come argomento essa viene salvata
            # in self.spins
            del kwargs['spins']  # dopo di che la eliminiamo
        else:  # se non diamo la catena iniziale la generiamo random
            size = 3
            if 'size' in kwargs:  # se diamo la lunghezza
                size = kwargs['size']
                del kwargs['size']
            random_seed = 1
            np.random.seed(random_seed)
            self.spins = 2 * np.random.randint(2, size=size) - 1
        assert len(self.spins) >= 3  # per vedere se tutto è ok verifichiamo che la lunghezza sia minimo >= 3

        if len(kwargs) > 0:
            raise TypeError(f'Unknown kwargs: {list(kwargs.keys())}')

    def deltaE(self, i):  # è una funzione
        """Return energy cost of flipping a given spin"""
        return 2 * self.spins[i] * (self.field + self.coupling * (self.spins[(i - 1) % len(self.spins)] +
                                                                  self.spins[(i + 1) % len(self.spins)]))

    def energy(self):  # funzione
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
        # create the object as before but adding the action rates.
        action_rates = (
            kwargs.pop('action_rates') if 'action_rates' in kwargs  # if in kwargs there is a key action_rates, we take
            else None  # its value and delete the key from the kwargs with pop
        )

        super().__init__(coupling, temperature, field, **kwargs)  # super().init allow me to go to the init of class
        # chain so that i don't have to rewrite the code. Alla fine ho dE, H (e il dizionario)

        self.action_rates = (
            np.full((len(self.spins), 2), 1) if action_rates is None
            else np.asarray(action_rates)
        )  # define the variable action rate
        assert self.action_rates.shape == (len(self.spins), 2)

        self._buffer = np.empty_like(self.spins)  # define the buffer

    def action_rate(self, i, value):
        """Return action rate for a given value of an `i`th spin (I need it for the step in Alg1)"""
        return self.action_rates[i, (value + 1) // 2]

    def _prepare_buffer(self):
        """Return buffer after ensuring that it has the required size"""
        if self._buffer.shape != self.spins.shape:
            self._buffer = np.empty_like(self.spins)
        return self._buffer

    def advance(self, index_criterion):
        """Apply one simulation step of the algorithm I"""
        buffer = self._prepare_buffer()
        buffer = np.full_like(buffer, 1) # RB: this kills the purpose of `_prepare_buffer()`
        # RB: Instead you should use np.fill(buffer, 1).

        # RB: Permuation is not necessary here, because we do not change anything
        # RB: until the pass is complete. `np.arange(...)` is enough.
        for spin_to_change in np.random.permutation(np.arange(len(self.spins))):
            dE = self.deltaE(spin_to_change)

            # RB: That is lame hard coding =) You have `action_rates` keyword argument (kwarg),
            # RB: which you can use to specify you action rates. Do not hard-code your action rates!
            if index_criterion:
                if spin_to_change % 2 != 0:
                    action_rates_ratio = self.action_rate(spin_to_change, 1) / self.action_rate(spin_to_change, -1)
                    weight = np.exp(-(1 / self.temperature) * dE) * action_rates_ratio
                    prob_change = self.action_rate(spin_to_change, -1) * weight / (1 + weight)
                else:
                    action_rates_ratio = self.action_rate(spin_to_change, -1) / self.action_rate(spin_to_change, 1)
                    weight = np.exp(-(1 / self.temperature) * dE) * action_rates_ratio
                    prob_change = self.action_rate(spin_to_change, 1) * weight / (1 + weight)
            else:
                if self.spins[spin_to_change] < 0:
                    action_rates_ratio = self.action_rate(spin_to_change, 1) / self.action_rate(spin_to_change, -1)
                    weight = np.exp(-(1 / self.temperature) * dE) * action_rates_ratio
                    prob_change = self.action_rate(spin_to_change, -1) * weight / (1 + weight)
                else:
                    action_rates_ratio = self.action_rate(spin_to_change, -1) / self.action_rate(spin_to_change, 1)
                    weight = np.exp(-(1 / self.temperature) * dE) * action_rates_ratio
                    prob_change = self.action_rate(spin_to_change, 1) * weight / (1 + weight)

            rank = np.random.random()
            if prob_change > rank:
                buffer[spin_to_change] = -1
        self.spins *= buffer
        return self.spins


class Metropolis(Chain):

    # definisco l'oggetto che e' quello di chain

    def __init__(self, coupling=1, temperature=1.0, field=0., **kwargs):
        super().__init__(coupling, temperature, field, **kwargs)

    def metropolis_pass(self):
        """Apply Metropolis step to each spin of the chain in random order

            Returns
                Chain: Updated chain
            """
        # Iterate over spin indices taken in the random order
        for spin_to_change in np.random.permutation(np.arange(len(self.spins))):

            # Metropolis condition min(1, exp(...)) always holds for dE < 0
            if np.random.random() < np.exp(- self.deltaE(spin_to_change) / self.temperature):
                self.spins[spin_to_change] *= -1
        return self.spins


def dynamic_evolution(chain: DynamicChain, time_evolution, correlation_flag, index_criterion):
    random_seed = 1
    np.random.seed(random_seed)
    engy = []
    m = []
    if correlation_flag:
        for t in tqdm(range(time_evolution), desc='Dynamic evolution'):
            engy.append(chain.energy())
            m.append(np.mean(chain.spins))
            chain.spins = chain.advance(index_criterion)
    else:
        initial_step = 0
        for t in tqdm(range(time_evolution), desc='Dynamic evolution'):
            if (t - initial_step) % 10 == 0:
                engy.append(chain.energy())
                m.append(np.mean(chain.spins))
                chain.spins = chain.advance(index_criterion)
                initial_step = t
    return engy, m


def acrl(m, time_evolution):
    m_demean = np.array([i - np.mean(m) for i in m])  # Demean the series
    # We are going to return a normalized autocorrelation function `c(t=0)=1`.
    nrm = np.sqrt(np.mean(pow(m_demean, 2)) * np.mean(pow(m_demean, 2)))
    tot = len(m_demean)
    m_corr = np.array([])
    for t in range(time_evolution):
        if nrm == 0:
            m_corr = np.append(m_corr, 1)
        else:
            m_corr = np.append(m_corr, [np.mean(m_demean[t:] * m_demean[:tot - t]) / nrm])

    return m_corr


def burn_in_evaluation(chain: Metropolis, time_evolution, correlation_flag):
    random_seed = 1
    np.random.seed(random_seed)
    # create the vector of trajectory starting for the initial state
    traj = []
    if correlation_flag:
        for _ in tqdm(range(time_evolution), desc='Burn-in evaluation'):
            # update the configuration over the time
            chain.spins = chain.metropolis_pass()
            # storage the value of the magnetization at time t
            traj.append(abs(np.mean(chain.spins)))
    else:
        initial_step = 0
        for t in tqdm(range(time_evolution), desc='Burn-in evaluation'):
            if (t - initial_step) % 10 == 0:
                # update the configuration over the time
                chain.spins = chain.metropolis_pass()
                # storage the value of the magnetization at time t
                traj.append(abs(np.mean(chain.spins)))
                initial_step = t

    fig = plt.figure()
    m_trajectory = [traj]

    plt.xlabel('Time steps', fontsize=14)
    plt.ylabel('Average magnetization', fontsize=14)
    fig = plt.plot(m_trajectory[0], label="Temperature={}".format(chain.temperature))
    plt.legend()
    plt.show()

    return fig


def correlation_time(chain: Metropolis, time_evolution, burn_in):
    random_seed = 1
    np.random.seed(random_seed)
    for t in range(burn_in):
        chain.spins = chain.metropolis_pass()
    m = np.array([])
    # first magnetization after burn in
    m = np.append(m, abs(np.mean(chain.spins)))
    for _ in tqdm(range(time_evolution), desc='Correlation time evaluation'):
        chain.spins = chain.metropolis_pass()
        # update next values of magnetization
        m = np.append(m, abs(np.mean(chain.spins)))

    m_corr = acrl(m=m, time_evolution=len(m))
    fig = plt.figure()
    m_trajectory = [m_corr]

    plt.xlabel('Time steps', fontsize=14)
    plt.ylabel('Temporal autocorrelation', fontsize=14)
    fig = plt.plot(m_trajectory[0], label="Temperature={}".format(chain.temperature))
    plt.legend()
    plt.show()
    return fig


def metropolis_ising(chain: Metropolis, n_samples, burn_in, corr_time, correlation_flag):
    random_seed = 1
    np.random.seed(random_seed)
    engy = []
    m = []
    for _ in tqdm(range(burn_in), desc='Waiting burn-in time'):
        chain.spins = chain.metropolis_pass()
    # primi valori di energia e magn dopo il burn - in
    engy.append(chain.energy())
    m.append(np.mean(chain.spins))
    if correlation_flag:
        samples = []
        for _ in tqdm(range(n_samples - 1), desc='Computing Montecarlo samples'):
            chain.spins = chain.metropolis_pass()
            samples.append(copy(chain.spins))
        # evaluate energy and magnetization over un-correlated sample
        for _ in tqdm(range(0, len(samples), corr_time), desc='Evaluation over un-correlated sample'):
            chain.spins = samples[_]
            m.append(np.mean(chain.spins))
            engy.append(chain.energy())
    else:
        if (n_samples - burn_in) % 2 != 0:
            n_samples = n_samples - burn_in + 1
        else:
            n_samples = n_samples - burn_in
        initial_step = 0
        for t in tqdm(range(n_samples), desc='Computing Montecarlo samples'):
            if (t - initial_step) % 10 == 0:
                chain.spins = chain.metropolis_pass()
                m.append(np.mean(chain.spins))
                engy.append(chain.energy())
                initial_step = t

    return engy, m


def theoretical_quantities(chain: Chain, n_samples):
    config = list(itertools.product([1, -1], repeat=len(chain.spins)))  # create 2^N strings of +1/-1
    theory_engy = []
    theory_m = []
    for conf in config:
        chain.spins = conf
        theory_m.append(np.mean(chain.spins))
        theory_engy.append(chain.energy())

    # RB: this can be majorly simplified
    weights_config = []
    theory_engy = np.sort(theory_engy)
    for value in theory_engy:
        weights_config.append(np.exp(-(1 / chain.temperature) * value))
    Z = sum(weights_config)
    config_prob = [weight / Z for weight in weights_config]
    keys = [float(value) for value in theory_engy]
    energy_prob = defaultdict(int)  # probability of a precise value of energy
    for k, n in zip(keys, config_prob):
        energy_prob[k] += n

    binomial_average = []
    binomial_std = []
    for i in range(len(list(energy_prob.values()))):
        binomial_average.append(scipy.stats.binom.mean(n=n_samples, p=list(energy_prob.values())[i]))
        binomial_std.append(scipy.stats.binom.std(n=n_samples, p=list(energy_prob.values())[i]))

    theory_engy_counts = []
    for engy_prob in list(energy_prob.values()):
        theory_engy_counts.append(engy_prob * n_samples)

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
    multiplicity = []
    std = []
    energy_level = defaultdict(int)
    keys = [float(value) for value in theory_engy]
    for k in zip(keys):
        energy_level[k] = 0
    i = 0
    for k in energy_level.keys():
        energy_level[k] = theory_avg[i]
        i += 1
    energy_level = {key[0]: value for key, value in energy_level.items()}
    # verificare che i livelli di energia coincidono con quelli teorici, aggiungi uno 0 per il livello mancante
    if len(list(energy_level.values())) > len(list(counts.values())):
        for i in range(len(list(energy_level.keys()))):
            if list(energy_level.keys())[i] not in list(counts.keys()):
                items = list(counts.items())
                items.insert(i, (list(energy_level.keys())[i], 0))
                counts = dict(items)
    counts = OrderedDict(counts)
    for i in range(len(theory_avg)):
        factor = abs(list(counts.values())[i] - theory_avg[i]) / theory_std[i]
        multiplicity.append(round(factor))
        if factor > 1:
            std.append(factor * theory_std[i])
        else:
            std.append(theory_std[i])

    return multiplicity, std, counts


def gof(f_obs, f_exp):
    t_statistics = 0
    for i in range(len(f_exp)):
        t_statistics += pow(f_obs[i] - f_exp[i], 2) / f_obs[i]

    # RB: remind me to discuss the degrees of freedom
    p_value = 1 - scipy.stats.chi2.cdf(x=t_statistics, df=len(f_exp) - 1)
    return p_value


def two_sample_chi2test(dict_a, dict_b, n_samples_a, n_samples_b):
    k1 = pow(n_samples_b / n_samples_a, 1 / 2)
    k2 = pow(n_samples_a / n_samples_b, 1 / 2)
    t_statistic = 0
    if len(dict_a) != len(dict_b):
        # RB: This is not enough. You may have the same length of your dictionaries,
        # RB: but the keys are not all equal, e.g. {'a': 1, 'b': 2}, {'a': 1, 'c': 3}.
        # RB: We should discuss the whole implementation—remind me please
        for i in range(len(list(dict_b.keys()))):
            if list(dict_b.keys())[i] not in list(dict_a.keys()):
                items = list(dict_a.items())
                items.insert(0, (list(dict_b.keys())[i], 0))
                dict_a = dict(items)
        dict_a = OrderedDict(sorted(dict_a.items()))

        for i in range(len(list(dict_a.keys()))):
            if list(dict_a.keys())[i] not in list(dict_b.keys()):
                items = list(dict_b.items())
                items.insert(0, (list(dict_a.keys())[i], 0))
                dict_b = dict(items)
        dict_b = OrderedDict(sorted(dict_b.items()))

    n_bins = len(list(dict_a.values()))
    df = n_bins - 1
    for bin in range(n_bins):
        if list(dict_a.values())[bin] == 0 and list(dict_b.values())[bin] == 0:
            df -= 1
        else:
            t_statistic += pow(k1 * list(dict_a.values())[bin] -
                               k2 * list(dict_b.values())[bin], 2) / (list(dict_a.values())[bin] +
                                                                      list(dict_b.values())[bin])
    p_value = scipy.stats.chi2.cdf(x=t_statistic, df=df)
    return p_value


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
    size = 1000
    T = 0.5
    h = 0
    J = 1
    action_rates = np.zeros((size, 2))
    index_criterion = False  # distinguish if action rates change by position(True) or by value(False)
    action_rates[:, 0] = 0.7
    action_rates[:, 1] = 0.7

    chain_metropolis = Metropolis(temperature=T, size=size, field=h, coupling=J)
    chain_dynamic = DynamicChain(temperature=T, size=size, field=h, coupling=J, action_rates=action_rates)

    n_samples = 10000
    correlation_flag = False  # distinguish two ways of de-correlate: if True generates all the samples and evaluate
    # H, M only at each t-correlation step. If false, generate a sample each 10 iterations (decrease sampling rate)

    fig_burnIn = burn_in_evaluation(chain_metropolis, time_evolution=n_samples, correlation_flag=correlation_flag)
    t_burnIn = int(input("Enter t_burn_in:"))
    if correlation_flag:
        fig_correlation = correlation_time(chain_metropolis, time_evolution=n_samples, burn_in=t_burnIn)
        t_correlation = int(input("Enter t_correlation:"))
    else:
        t_correlation = 0

    metropolis_engy, metropolis_m = metropolis_ising(chain_metropolis, n_samples=n_samples, burn_in=t_burnIn,
                                                     corr_time=t_correlation, correlation_flag=correlation_flag)
    dynamic_engy, dynamic_m = dynamic_evolution(chain_dynamic, time_evolution=n_samples,
                                                correlation_flag=correlation_flag, index_criterion=index_criterion)

    metropolis_m_counts = count_variables(metropolis_m)
    dynamic_m_counts = count_variables(dynamic_m)
    metropolis_engy_counts = count_variables(metropolis_engy)
    dynamic_engy_counts = count_variables(dynamic_engy)

    if size <= 4:

        if not correlation_flag:
            n_samples = n_samples / 10

        theory_engy, theory_m, theory_engy_counts, binomial_avg, binomial_std = theoretical_quantities(chain_metropolis,
                                                                                                       n_samples=n_samples)

        multiplicity_metropolis, std_metropolis, metropolis_engy_counts = std_algorithms(metropolis_engy_counts,
                                                                                         binomial_avg, theory_engy,
                                                                                         binomial_std)
        multiplicity_dynamic, std_dynamic, dynamic_engy_counts = std_algorithms(dynamic_engy_counts, binomial_avg,
                                                                                theory_engy, binomial_std)
        print(binomial_std, multiplicity_metropolis, multiplicity_dynamic)
        fig, ax = plt.subplots()
        labels = metropolis_engy_counts.keys()
        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars
        rects1 = ax.bar(x, list(metropolis_engy_counts.values()), width, yerr=std_metropolis, align='center',
                        label='MonteCarlo', color='m')
        rects2 = ax.bar(x + width, list(dynamic_engy_counts.values()), width, yerr=std_dynamic, align='center',
                        label='Dynamic', color='b')
        theory_engy_counts_adjusted = [round(counts) for counts in theory_engy_counts]
        rects3 = ax.bar(x + width * 2, theory_engy_counts_adjusted, width, yerr=binomial_std, align='center',
                        label='Theory', color='g')

        ax.set_title('Bar Chart')
        ax.set_ylabel('Counts per energy level')
        ax.set_xlabel('Energy level')
        ax.set_xticks(x + width, labels)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        fig.tight_layout()
        plt.show()

        '''
        goodness of fit test (one-sample vs theory):
        test brings out problems for very low or very high T because in these case might happen that the
        theory_engy_counts there exists a cell with almost 0 counts in the t-statistic so we're going to divide per zero
        '''
        metropolis_pvalue = gof(f_obs=list(metropolis_engy_counts.values()), f_exp=theory_engy_counts)
        dynamic_pvalue = gof(f_obs=list(dynamic_engy_counts.values()), f_exp=theory_engy_counts)

        print(metropolis_pvalue, dynamic_pvalue)

    else:

        fig_kde, axes = plt.subplots(1, 2)
        h1 = sns.kdeplot(data=metropolis_engy, bw_method='scott', color='m', ax=axes[0])
        h1.set(xlabel=None)
        h2 = sns.kdeplot(data=dynamic_engy, bw_method='scott', color='b', ax=axes[0])
        h2.set(xlabel=None)
        m1 = sns.kdeplot(data=metropolis_m, bw_method='scott', color='m', ax=axes[1])
        m1.set(xlabel=None)
        m2 = sns.kdeplot(data=dynamic_m, bw_method='scott', color='b', ax=axes[1])
        m2.set(xlabel=None)

        axes[0].set_title("Energy")
        axes[0].legend(['MonteCarlo', 'Dynamic'], loc="upper left")
        axes[1].set_title("Magnetization")
        axes[1].legend(['MonteCarlo', 'Dynamic'], loc="upper right")

        bin_edges_engy_mc, metropolis_engy = hist(chain_metropolis, metropolis_engy, engy_flag=True)
        bin_edges_engy_dynamic, dynamic_engy = hist(chain_dynamic, dynamic_engy, engy_flag=True)
        fig_engy, axes = plt.subplots(1, 2)
        axes[0].hist(metropolis_engy, bins=bin_edges_engy_mc, label='Metropolis', color='m')
        axes[1].hist(dynamic_engy, bins=bin_edges_engy_dynamic, label='Dynamic', color='b')
        axes[0].set_title("Energy Montecarlo")
        axes[1].set_title("Energy Dynamic")
        bin_edges_m_mc, metropolis_m = hist(chain_metropolis, metropolis_m, engy_flag=False)
        bin_edges_m_dynamic, dynamic_m = hist(chain_dynamic, dynamic_m, engy_flag=False)
        fig_m, axes = plt.subplots(1, 2)
        axes[0].hist(metropolis_m, bins=bin_edges_m_mc, label='Metropolis', color='m')
        axes[1].hist(dynamic_m, bins=bin_edges_m_dynamic, label='Dynamic', color='b')
        axes[0].set_title("Magnetization Montecarlo")
        axes[1].set_title("Magnetization Dynamic")
        plt.show()

    # two samples chi2test
    engy_pvalue = two_sample_chi2test(metropolis_engy_counts, dynamic_engy_counts, n_samples_a=len(metropolis_engy),
                                      n_samples_b=len(dynamic_engy))
    m_pvalue = two_sample_chi2test(metropolis_m_counts, dynamic_m_counts, n_samples_a=len(metropolis_m),
                                   n_samples_b=len(dynamic_m))

    print(engy_pvalue, m_pvalue)
