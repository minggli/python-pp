#! /usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings

import pandas as pd

import pymc3 as pm
import theano.tensor as tt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')
mpl.rcParams['figure.dpi'] = 150
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


mat = pd.read_csv('./data/student-mat.csv', sep=';')
mat.drop(['G2', 'G3'], axis=1, inplace=True)
mat = mat.select_dtypes(exclude=['object'])

# EDA
print(mat.head().transpose())
mat.info(verbose=True, memory_usage='deep', null_counts=True)

encoded_mat, categorical_mapping = encode_categorical(mat)
features = encoded_mat[encoded_mat.columns.difference(['G1'])]
label = encoded_mat[['G1']]

# feature selection, 5 continuous random variables only for this exercise
bck_select = RFE(LinearRegression(), 5, 1)
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
# plt.show()


model = pm.Model()
with model:
    # bayesian linear regression
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ_vector = pm.Normal('θ', mu=0, sd=10, shape=5)
    μ = pm.Deterministic('μ', intercept + tt.dot(subset_mat, θ_vector))
    σ = pm.HalfNormal('σ', sd=10)

    y_obs = pm.Normal('y_obs', mu=μ, sd=σ, observed=label)
    trace = pm.sample(500)
