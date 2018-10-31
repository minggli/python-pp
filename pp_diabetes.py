#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

# np.random.seed(0)

X, y = load_diabetes(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = X[:, :3]
y = np.log(y)

_, n = X.shape
lr = LinearRegression(fit_intercept=True)
lr.fit(X, y)
print(f"OLS estimated intercept: {lr.intercept_}, betas: {lr.coef_}")

model = pm.Model()

with model:
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ = pm.Normal('θ', mu=0, sd=10, shape=(n))
    μ = intercept + pm.math.dot(X, θ)
    # TODO dot product doesn't seem to work
    σ = pm.HalfNormal('σ', sd=10)
    y_obs = pm.Normal('yhat', mu=μ, sd=σ, observed=y)


print('finished specifying model:')
print(model.check_test_point())

with model:
    sampler = pm.Metropolis()
    print("Sampling using Metropolis-Hastings:")
    trace = pm.sample(5000, tuning=5000, step=sampler)

with model:
    map_estimates = pm.find_MAP()
print(f"Maximum-a-Posteriori estimates: {map_estimates}")

print("Automatic Differentation Variational Inference")
with model:
    inference = pm.ADVI()
    approx = pm.fit(30000, method=inference)
    vi_trace = approx.sample(draws=5000)

pm.traceplot(trace)
pm.traceplot(vi_trace)
plt.show()
