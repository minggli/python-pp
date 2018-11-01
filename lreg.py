#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
import warnings

import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3.distributions.discrete import Bernoulli

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

mpl.style.use('ggplot')
warnings.simplefilter('ignore')

X, y = load_breast_cancer(return_X_y=True)
X = X[:, :3]
X = StandardScaler().fit_transform(X)
m, n = X.shape

model = pm.Model()

with model:
    intercept = pm.Normal('intercept', mu=0, sd=10)
    θ = pm.Normal('θ', mu=0, sd=1, shape=(n))
    z = intercept + pm.math.dot(X, θ)
    logit = pm.math.sigmoid(z)
    likelihood = pm.Bernoulli('yhat', logit_p=logit, observed=y)

with model:
    step = pm.NUTS()
    trace = pm.sample(3000, tuning=1000, step=step)

lreg = LogisticRegression()
lreg.fit(X.reshape(-1, n), y)
print(f"logit: {lreg.intercept_}, betas: {lreg.coef_}")

pm.traceplot(trace)
plt.show()
