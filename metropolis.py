"""
Metropolis-Hastings

naive implementation of MH Sampling.

Reference:
Yildirim Ilker 2012. Bayesian Inference: Metropolis-Hastings Sampling.
Available at: http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf
"""
from typing import Callable

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from tqdm import tqdm

sns.set()


def simple_p(x):
    return stats.norm.pdf(x, 30, 10)


def dummy_p(x):
    val = stats.norm.pdf(x, loc=30, scale=10)
    val += stats.norm.pdf(x, loc=80, scale=20)
    return val


def metropolis_hastings(p: Callable, n_iter: int = 1000, burn_in: int = 200):
    k = 1
    # initial sample.
    X_t_1 = np.zeros(k)

    # container for samples from discrete markov process
    samples = np.zeros((n_iter, k))
    mu_arr = np.random.rand(n_iter)

    for i in tqdm(range(n_iter), desc="Metropolis Hastings"):
        # MH criterion:
        # [p(X_t) * q(X_t_1|X_t)] / [p(X_t_1) * q(X_t|X_t_1)]
        # Gaussian has symmetricity so q(X_t_1|X_t) = q(X_t|X_t_1)
        # so using Gaussian proposal, MH reduces to Metropolis algorithm.

        # draw from proposal distribution formed of Markov Chain: q(x_t|x_t_1)
        X_t = X_t_1 + np.random.normal(scale=1)

        r_acceptance = min(1, p(X_t) / p(X_t_1))
        if r_acceptance > mu_arr[i]:
            X_t_1 = X_t

        samples[i] = X_t_1

    if burn_in:
        return samples[int(burn_in):]
    else:
        return samples


if __name__ == "__main__":

    target_p = dummy_p
    simulation = metropolis_hastings(target_p, n_iter=50000, burn_in=5000)

    X = np.linspace(0, 200, 15000)
    true_samples = list(map(target_p, X))

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    sns.distplot(simulation, ax=ax[0])
    ax[1].plot(X, true_samples)
    plt.show()
