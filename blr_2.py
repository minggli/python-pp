#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings

import pandas as pd
import numpy as np

import pymc3 as pm

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ols import ols

warnings.simplefilter('ignore')
mpl.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')


def encode_categorical(dataframe):
    """encode categorical more than two categories."""
    bool_mask = dataframe.select_dtypes(['object']).nunique() <= 2
    cats = dataframe.select_dtypes(['object']).loc[:, ~bool_mask].columns

    mapping = dict()
    for col in dataframe.select_dtypes(['object']).loc[:, bool_mask]:
        dataframe[col] = dataframe[col].astype('category')
        mapping[col] = dict(enumerate(dataframe[col].cat.categories))
        dataframe[col] = dataframe[col].cat.codes

    return pd.get_dummies(dataframe, columns=cats), mapping


def predict(X, trace):
    return trace['intercept'][:, np.newaxis] + trace['θ'] @ X.T


def ols_predict(X, intercept, theta):
    return intercept + X @ theta


mat = pd.read_csv('./data/student-mat.csv', sep=';')
mat.drop(['G2', 'G3'], axis=1, inplace=True)
mat = mat.select_dtypes(exclude=['object'])

# EDA
# mat.info(verbose=True, memory_usage='deep', null_counts=True)

encoded_mat, categorical_mapping = encode_categorical(mat)
features = encoded_mat[encoded_mat.columns.difference(['G1'])]
label = encoded_mat[['G1']]

# feature selection, 5 continuous random variables only for this exercise
n_var = 2
bck_select = RFE(LinearRegression(), n_var, 1)
bck_select.fit(features, label)
ranks = tuple(zip(features, bck_select.ranking_))
subset_mat = features.loc[:, bck_select.support_]
print(subset_mat.head())

fig = plt.figure(figsize=(10, 10), dpi=100)
ax1 = Axes3D(fig)
ax1.scatter(encoded_mat['studytime'], encoded_mat['failures'],
            encoded_mat['G1'])
ax1.set_xlabel('study time')
ax1.set_ylabel('failures')
ax1.set_zlabel('G1 score')

X_train, X_test, y_train, y_test = \
    train_test_split(subset_mat, label, train_size=.8, test_size=.2)

model = pm.Model()
with model:
    # bayesian linear regression
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ_vector = pm.Normal('θ', mu=0, sd=10, shape=n_var)
    μ = intercept + pm.math.dot(X_train, θ_vector[:, np.newaxis])
    σ = pm.HalfNormal('σ', sd=10)
    # y ~ N(intercept + θ.T @ X, σ)
    y_obs = pm.Normal('y_obs', mu=μ, sd=σ, observed=y_train)
    posterior = pm.sample(1000, tune=1000)
    try:
        pm.traceplot(posterior)
    except AttributeError:
        pass
    pm.plot_posterior(posterior)

plt.show()

# prediction
yhat = predict(X_test, posterior).T

ols_intercept, ols_theta = ols(X_train, y_train)
ols_yhat = ols_predict(X_test, ols_intercept, ols_theta)

xgr = XGBRegressor()
xgr.fit(X_train, y_train)
xgr_yhat = xgr.predict(X_test)

for i in range(3):
    n = np.random.randint(0, y_test.shape[0])
    sns.kdeplot(yhat[n], label='Bayesian Posterior Predictive_{}'.format(n))
    plt.vlines(x=ols_yhat[n], ymin=0, ymax=10,
               label='manual OLS Prediction_{}'.format(n), colors='blue',
               linestyles='--')
    plt.vlines(x=y_test.values[n], ymin=0, ymax=10,
               label='Actual_{}'.format(n), colors='red', linestyles='-')
    plt.vlines(x=xgr_yhat[n], ymin=0, ymax=10,
               label='Xgboost_{}'.format(n), colors='black',
               linestyles='--')
    plt.legend(loc='upper left')
    plt.show()

base_mse = mean_squared_error(
                y_test, np.array([y_train.mean()] * y_test.shape[0]))
print('Baseline (uniform assumption) produced test set Mean Squared Error: '
      '{0:.4f}'.format(base_mse))
ols_mse = mean_squared_error(y_test, ols_yhat)
print('OLS produced test set Mean Squared Error: {0:.4f}'.format(ols_mse))
xgr_mse = mean_squared_error(y_test, xgr_yhat)
print('Xgboost produced test set Mean Squared Error: {0:.4f}'.format(xgr_mse))
mse = mean_squared_error(y_test, yhat.mean(axis=1))
print('BLR produced test set Mean Squared Error: {0:.4f}'.format(mse))
