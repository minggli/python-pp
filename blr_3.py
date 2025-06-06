#! /usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import copy

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import pymc as pm


def subsample(array_like, s, replace=True):
    m, n = array_like.shape
    if hasattr(array_like, 'index'):
        index = array_like.index
    else:
        index = np.arrange(len(array_like))

    subsampled_index = np.random.choice(index, s, replace=replace)

    try:
        # index-based
        return copy(array_like.loc[subsampled_index])
    except AttributeError:
        # positional
        return copy(array_like[subsampled_index])


# np.random.seed(0)

raw = pd.read_csv('./data/exercise.csv').merge(
        pd.read_csv('./data/calories.csv')).drop(['User_ID'], axis=1)
df = subsample(raw, 10000, replace=False)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'male' else 0).astype(np.uint8)

X_train = df.loc[:, df.columns.difference(['Calories', 'Gender'])]
X = df.loc[:, ['Age', 'Duration']]
y = df.loc[:, ['Calories']]

_, n = X.shape

lr = LinearRegression(fit_intercept=True)
lr.fit(X, y)
print(f"OLS estimated intercept: {lr.intercept_}, betas: {lr.coef_}")

model = pm.Model()
# Old codes

with model:
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    θ = pm.Normal('θ', mu=0, sigma=10, shape=n)
    # μ = intercept + pm.math.dot(X.values, θ)
    # TODO dot product doesn't seem to work
    μ = intercept
    for k in range(n):
        μ += θ[k] * X.values[:, k].reshape(-1, 1)
    σ = pm.HalfNormal('σ', sigma=5)
    y_obs = pm.Normal('yhat', mu=μ, sigma=σ, observed=y)


print('finished specifying model:')
print(model.check_test_point())

with model:
    sampler = pm.Metropolis()
    print("Sampling using Metropolis-Hastings:")
    trace = pm.sample(10000, tuning=500, step=sampler)

with model:
    map_estimates = pm.find_MAP()
print(f"Maximum-a-Posteriori estimates: {map_estimates}")

print("Automatic Differentation Variational Inference")
with model:
    inference = pm.ADVI()
    approx = pm.fit(50000, method=inference)
    vi_trace = approx.sample(draws=10000)

pm.plot_trace(vi_trace)
pm.plot_trace(trace)
plt.show()
