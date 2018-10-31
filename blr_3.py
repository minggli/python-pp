#! /usr/bin/env python
# -*- encoding: utf-8 -*-

from copy import copy

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import pymc3 as pm


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


np.random.seed(0)

raw = pd.read_csv('./data/exercise.csv').merge(
        pd.read_csv('./data/calories.csv')).drop(['User_ID'], axis=1)
df = subsample(raw, 500, replace=False)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'male' else 0).astype(np.uint8)

X_train = df.loc[:, df.columns.difference(['Calories', 'Gender'])]
X_train = df.loc[:, ['Duration']]
y_train = df.loc[:, ['Calories']]

_, n = X_train.shape

lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
print(f"OLS estimated intercept: {lr.intercept_}, betas: {lr.coef_}")

with pm.Model() as blr:
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ = pm.Normal('θ', mu=0, sd=10, shape=n)
    μ = intercept + pm.math.dot(X_train, θ)
    σ = pm.HalfNormal('σ', sd=10)
    y_obs = pm.Normal('y_obs', mu=μ, sd=σ, observed=y_train)

print('finished specifying model')

with blr:
    map_estimates = pm.find_MAP()
print(f"Maximum-a-Posteriori estimates: {map_estimates}")

print("Automatic Differentation Variational Inference")
with blr:
    inference = pm.ADVI()
    approx = pm.fit(20000, method=inference)
    vi_trace = approx.sample(draws=5000)

print("Sampling methods using Metropolis-Hastings")
with blr:
    sampler = pm.Metropolis()
    trace = pm.sample(5000, tuning=500, step=sampler)

pm.traceplot(vi_trace)
pm.traceplot(trace)
plt.show()

