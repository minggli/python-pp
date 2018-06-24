#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import warnings

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
import pymc3 as pm

warnings.simplefilter('ignore')
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['figure.figsize'] = 5, 10
plt.style.use('ggplot')
np.random.seed(0)


def resample(dataframe, s=500):
    m, n = dataframe.shape
    assert s < m
    subset = np.random.choice(m, s, replace=False)
    return dataframe.loc[subset].copy(deep=False).reset_index(drop=True)


def ols(X, y):
    # X @ B = y
    # X.T @ X @ B = X.T @ y
    # QR decomposition of X and avoid covaraince matrix inversion.
    # (Q @ R).T @ (Q @ R) @ B = X.T @ y
    # R.T @ (Q.T @ Q) @ R @ B = X.T @ y
    # Q is orthogonal so Q.T = inv(Q) and Q.T @ Q = I
    # R.T @ R @ B = R.T @ Q.T @ y
    # R @ B = Q.T @ y
    # R is upper triangle so solve using back subsubstitution or invert R
    # inverting:
    # R**-1 @ R @ B = R**-1 @ Q.T @ y
    # B = R**(-1) @ Q.T @ y
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X_offset, y_offset = np.mean(X), np.mean(y)
    X -= X_offset
    y -= y_offset
    q, r = np.linalg.qr(X)
    assert np.allclose(X, q @ r)
    theta = np.linalg.inv(r) @ q.T @ y
    intercept = y_offset - np.dot(X_offset, theta.T)
    return intercept, theta


def plot_ols(X, intercept, theta):
    X = np.asarray(X).reshape(-1, 1)
    return intercept + np.dot(X, theta)


raw = pd.read_csv('./data/exercise.csv').merge(
            pd.read_csv('./data/calories.csv')).drop(['User_ID'], axis=1)

df = resample(raw, s=100)
X = df['Duration'].copy()
y = df['Calories'].copy()

lr = LinearRegression()
lr.fit(X.to_frame(), y)
base = np.linspace(1, X.max(), X.max())
intercept, theta = ols(X, y)
# check if manual OLS estimate is same as framework estimate
assert np.isclose(intercept, lr.intercept_)
assert np.isclose(theta, lr.coef_)

yhat = plot_ols(base, intercept, theta)

sns.lmplot('Duration', 'Calories', df, fit_reg=False)
plt.plot(base, yhat, color='black', linestyle='dashed',
         label='Ordinary Least Square')
plt.legend()
plt.show()

model = pm.Model()
