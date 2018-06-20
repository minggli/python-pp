#! /usr/bin/env python
# -*- encoding: utf-8 -*-
"""
bayesian linear regression (motivation example from PyMC3)
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
plt.style.use('ggplot')

# Initialize random number generator
np.random.seed(0)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

# plotting
# fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
# axes[0].scatter(X1, Y)
# axes[1].scatter(X2, Y)
# axes[0].set_ylabel('Y')
# axes[0].set_xlabel('X1')
# axes[1].set_xlabel('X2')
# plt.show()

# modelling
model = pm.Model()
with model:
    # Priors P(alpha, beta, sigma) for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome, 𝘶 = alpha + B * X
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    # P(y|𝘶, 𝜎)
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

# maximum a posteriori only provides point estimate of parameters?
# TODO instead of posterior distribution of theta?
map_estimate = pm.find_MAP(model=model)
print(map_estimate)
# sampling methods
with model:
    # draw 500 posterior samples
    trace = pm.sample(500)

pm.traceplot(trace)
plt.show()
