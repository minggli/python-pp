"""
rejection sampling

multiprocessing rejection sampler
"""

import sys
import multiprocessing
from multiprocessing import Value, Process, Manager, cpu_count, current_process

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm, lognorm
from tqdm import tqdm
from numba import jit


def maybe_decorate(func):
    try:
        f = jit(func,
                nopython=False,
                fastmath=True,
                parallel=False,
                forceobj=True)
    except AssertionError:
        f = jit(func)
    except Exception:
        f = func
    finally:
        return f


def unimodal_p(x):
    return norm.pdf(x, loc=30, scale=10)


def asymmetric_p(x):
    return lognorm.pdf(x, .5, loc=30, scale=10)


def bimodal_p(x):
    val = norm.pdf(x, loc=30, scale=10)
    val += norm.pdf(x, loc=80, scale=20)
    return val


def rejection_sampling(func,
                       *args,
                       sample_size=5000,
                       support=[0, 1],
                       envelop=norm.pdf):
    """multiprocessing rejection sampler."""

    def _sample(ssize, k, samples):
        """rejection sampling subroutine"""
        pos = processes.index(current_process())
        pbar = tqdm(desc=f"CPU {pos}", total=-(-ssize // c), position=pos)
        while len(samples) < ssize:
            z = np.random.uniform(*support)
            y_hat = k.value * envelop(z, *support)
            u = np.random.uniform(0, y_hat)
            y = func(z, *args)
            if y > y_hat:
                # elevate envelop distribution to cover target distribution.
                k.value += 1
            if u <= y:
                samples.append((z, u))
                pbar.update(1)
        pbar.close()

    c = cpu_count()
    # shared scaling variable k for all processes
    k = Value('i', 3)
    samples = Manager().list()
    func = maybe_decorate(func)

    processes = []
    for core in range(c):
        p = Process(target=_sample, args=(sample_size, k, samples))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    return samples[:sample_size]


def inverse_sampling(quantile_func, *params, sample_size=100, sort=True):
    """sample inverse of cumulative distribution function of domain (0, 1)
    back to the original distribution of random variable X. """

    quantile_func = maybe_decorate(quantile_func)
    cdf = np.random.uniform(0, 1, sample_size)
    if sort:
        cdf.sort()
    sys.stdout.write("sampling: {0}\n".format(repr(quantile_func)))
    sys.stdout.flush()
    _x = (quantile_func(x, *params) for x in tqdm(cdf, total=sample_size))
    x = np.fromiter(_x, np.float32)
    return cdf, x


if __name__ == "__main__":
    sns.set_theme()
    multiprocessing.set_start_method('fork')

    target_p = bimodal_p
    samples = rejection_sampling(target_p,
                                 support=[0, 200],
                                 sample_size=20000)
    x = np.linspace(0, 200, 20000)
    true_samples = np.fromiter((target_p(i) for i in x), dtype=np.float32)
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(12, 9))
    ax[0].set_title("Rejection Sampling")
    sns.histplot(list(zip(*samples))[0], ax=ax[0])
    # ax[0].scatter(list(zip(*samples))[0], list(zip(*samples))[1], s=0.1)
    ax[1].set_title("Probability Density Function")
    ax[1].plot(x, true_samples)
    plt.savefig("rejection_sampling_and_pdf.png")
