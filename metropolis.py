"""
Metropolis-Hastings

naive implementation of MH Sampling.

Reference:
Yildirim Ilker 2012. Bayesian Inference: Metropolis-Hastings Sampling.
Available at: http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf
"""
from typing import Callable

import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from tqdm import tqdm


def unimodal_p(x):
    return stats.norm.logpdf(x, 30, 10)


def bimodal_p(x):
    val = stats.norm.pdf(x, loc=30, scale=10)
    val += stats.norm.pdf(x, loc=80, scale=20)
    return np.log(val)


def asymmetric_p(x):
    return stats.lognorm.logpdf(x, .5, loc=30, scale=10)


def metropolis_hastings(p: Callable, n_iter: int = 1000, burn_in: int = 200):
    k = 1
    # initial sample.
    X_t_1 = np.zeros(k)

    # container for samples from discrete markov process
    samples = np.zeros((n_iter, k))
    mu_arr = np.log(np.random.rand(n_iter))

    for t in tqdm(range(n_iter), desc="Metropolis Hastings"):
        # MH criterion:
        # [p(X_t) * q(X_t_1|X_t)] / [p(X_t_1) * q(X_t|X_t_1)]
        # Gaussian has symmetricity so q(X_t_1|X_t) = q(X_t|X_t_1)
        # so using Gaussian proposal, MH reduces to Metropolis algorithm.

        # draw from proposal distribution centered t_1 in Markov Chain: q(x_t|x_t_1)
        X_t = X_t_1 + np.random.normal()

        # acceptance ratio in log scale
        r_acceptance = min(0, p(X_t) - p(X_t_1))
        if r_acceptance > mu_arr[t]:
            X_t_1 = X_t

        samples[t] = X_t_1

    if burn_in:
        return samples[int(burn_in):]
    else:
        return samples


def handle_args(args) -> Callable:
    if args.d is None:
        return asymmetric_p
    else:
        if "unimodal" in args.d:
            return unimodal_p
        elif "bimodal" in args.d:
            return bimodal_p
        elif "asym" in args.d:
            return asymmetric_p
        else:
            print(f"unrecognized argument value {args.d} for --d")
            return asymmetric_p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=str)
    parser.add_argument("--iter", type=int)
    args = parser.parse_args()

    sns.set_theme()

    n_iter = args.iter or 50000
    target_p = handle_args(args)
    samples = metropolis_hastings(target_p,
                                  n_iter=n_iter,
                                  burn_in=min(2000, n_iter // 10))
    X = np.linspace(-50, 150, 15000)
    y = np.exp(list(map(target_p, X)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    axes[0].set_title("Markov Chain")
    axes[0].plot(samples)
    axes[1].set_title("MCMC Sampling")
    sns.histplot(samples, ax=axes[1], legend=False)
    axes[2].sharex(axes[1])
    axes[2].set_title("Probability Density Function")
    axes[2].plot(X, y)
    plt.tight_layout()
    plt.savefig("MCMC_sampling_and_pdfs.png")
