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

from ols import ols

warnings.simplefilter('ignore')
mpl.rcParams['figure.dpi'] = 100
mpl.style.use('ggplot')
# np.random.seed(0)


def resample(dataframe, s=500):
    m, n = dataframe.shape
    assert s < m
    subset = np.random.choice(m, s, replace=False)
    return dataframe.loc[subset].copy(deep=False).reset_index(drop=True)


def predict_ols(X, intercept, theta):
    X = np.asarray(X).reshape(-1, 1)
    return intercept + np.dot(X, theta)


def bayesian_predictive_distribution(x, trace):
    return trace['theta'] * x + trace['intercept']


raw = pd.read_csv('./data/exercise.csv').merge(
            pd.read_csv('./data/calories.csv')).drop(['User_ID'], axis=1)

df = resample(raw, s=50)
X = df['Duration']
y = df['Calories']

lr = LinearRegression()
lr.fit(X.to_frame(), y)
base = np.linspace(1, X.max(), 100)
intercept, theta = ols(X, y)
# check if manual OLS estimate is same as framework estimate
assert np.isclose(intercept, lr.intercept_)
assert np.isclose(theta, lr.coef_)

yhat = predict_ols(base, intercept, theta)

model = pm.Model()
with model:
    std = pm.HalfNormal('standard deviation', sd=10)
    y_obs = pm.Normal('yhat',
                      mu=(pm.Normal('intercept', mu=0, sd=10) +
                          pm.Normal('theta', mu=0, sd=10) * X),
                      sd=std,
                      observed=y.values)
    step = pm.NUTS()
    trace = pm.sample(1000, step)

sns.lmplot('Duration', 'Calories', df, fit_reg=False)
pm.plot_posterior_predictive_glm(
    trace, samples=100, eval=base, linewidth=.3, color='r', alpha=0.8,
    label='Bayesian Posterior Predictive',
    lm=lambda x, sample: sample['intercept'] + sample['theta'] * x)
plt.plot(base, yhat, color='black', linestyle='dashed',
         label='Ordinary Least Square')
plt.legend()
plt.show()

# Prediction
x = 15
ols_yhat = predict_ols(x, intercept, theta)
bayes_pred = bayesian_predictive_distribution(x, trace)
sns.kdeplot(bayes_pred, label='Bayes Posterior Predictive Distribution')
plt.vlines(x=ols_yhat, ymin=0, ymax=2.5,
           label='OLS Prediction', colors='red', linestyles='--')
plt.legend(loc='upper left')
plt.show()
