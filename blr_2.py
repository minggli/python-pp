#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings

import pandas as pd
import numpy as np

import pymc3 as pm
import theano.tensor as tt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
mpl.rcParams['figure.dpi'] = 100
plt.style.use('ggplot')

np.random.seed(0)


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


mat = pd.read_csv('./data/student-mat.csv', sep=';')
mat.drop(['G2', 'G3'], axis=1, inplace=True)
mat = mat.select_dtypes(exclude=['object'])

# EDA
mat.info(verbose=True, memory_usage='deep', null_counts=True)

encoded_mat, categorical_mapping = encode_categorical(mat)
features = encoded_mat[encoded_mat.columns.difference(['G1'])]
label = encoded_mat[['G1']]

# feature selection, 5 continuous random variables only for this exercise
bck_select = RFE(LinearRegression(), 5, 1)
bck_select.fit(features, label)
ranks = tuple(zip(features, bck_select.ranking_))
subset_mat = features.loc[:, bck_select.support_]

fig = plt.figure(figsize=(10, 10), dpi=100)
ax1 = Axes3D(fig)
ax1.scatter(encoded_mat['studytime'], encoded_mat['failures'],
            encoded_mat['G1'])
ax1.set_xlabel('study time')
ax1.set_ylabel('failures')
ax1.set_zlabel('G1 score')
# plt.show()

X_train, X_test, y_train, y_test = \
    train_test_split(subset_mat, label, train_size=.8, test_size=.2)

model = pm.Model()
with model:
    # bayesian linear regression
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ_vector = pm.Normal('θ', mu=0, sd=10, shape=5)
    μ = intercept + tt.dot(X_train, θ_vector)
    σ = pm.HalfNormal('σ', sd=10)

    y_obs = pm.Normal('y_obs', mu=μ, sd=σ, observed=y_train)
    vi = pm.fit()
    posterior = vi.sample(1000)
    pm.plot_posterior(posterior)

plt.show()
