import sys
from multiprocessing import Value, Process, Manager, cpu_count, current_process

import numpy as np

from scipy.stats import norm
from tqdm import tqdm
from numba import jit


def maybe_decorate(func):
    try:
        f = jit(func, nopython=True, fastmath=True, parallel=True)
    except AssertionError:
        f = jit(func)
    except:
        f = func
    finally:
        return f


def inverse_sampling(quantile_func, *params, sample_size=100, sort=True):
    """sample inverse of cumulative distribution function of domain (0, 1)
    back to the original distribution of random variable X."""

    quantile_func = maybe_decorate(quantile_func)
    cdf = np.random.uniform(0, 1, sample_size)
    if sort:
        cdf.sort()
    sys.stdout.write("sampling: {0}\n".format(repr(quantile_func)))
    sys.stdout.flush()
    _x = (quantile_func(x, *params) for x in tqdm(cdf, total=sample_size))
    x = np.fromiter(_x, np.float32)
    return cdf, x


def rejection_sampling(func,
                       *args,
                       sample_size=1000,
                       support=[0, 1],
                       proposal=norm):

    def _sample(ssize, k, samples):
        pos = processes.index(current_process())
        for _ in tqdm(range(ssize), position=pos):
            z = np.random.uniform(*support)
            y_hat = k.value * proposal.pdf(z, *support)
            u = np.random.uniform(0, y_hat)
            y = func(z, *args)
            if y > y_hat:
                k.value += 1
            if u <= y:
                samples.append((z, u))

    c = cpu_count()
    # shared scaling variable k for all processes
    k = Value('i', 3)
    samples = Manager().list()
    per_process_samples = -(-sample_size // c)

    func = maybe_decorate(func)

    processes = []
    for core in range(c):
        p = Process(target=_sample, args=(per_process_samples, k, samples))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    return samples[:sample_size]
