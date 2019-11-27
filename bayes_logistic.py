#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simple script for Bayesian Logistic Regression
"""
from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pyro
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn

from pyro.optim import Adam
from pyro.infer import Trace_ELBO, SVI, TracePredictive, EmpiricalMarginal
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Normal, Bernoulli, Delta

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 100


class TorchLogisticRegression(nn.Module):
    def __init__(self, p):
        super(TorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(p, 1)
        self.nonlinearity = nn.Sigmoid()

    def logit(self, x):
        return self.linear(x)

    def forward(self, z):
        return self.nonlinearity(self.logit(z))


def seperate_posteriors(samples_dict, new_sites):
    new_samples_dict = {}
    for site in samples_dict:
        posterior_samples = samples_dict[site]
        dim = posterior_samples.shape[-1]
        if dim > 1:
            assert dim == len(new_sites)
            for i in range(dim):
                new_samples_dict[new_sites[i]] = posterior_samples[:, :, i]
        else:
            new_samples_dict["bias"] = posterior_samples[:, :, 0]
    return new_samples_dict


def bayes_logistic(X, y):
    n, k = X.shape
    w_prior = Normal(torch.zeros(1, k), torch.ones(1, k)).to_event(1)
    b_prior = Normal(torch.tensor([[0.]]), torch.tensor([[10.]])).to_event(1)
    priors = {"linear.weight": w_prior, "linear.bias": b_prior}
    lifted_module = \
        pyro.random_module("bayes_logistic", frequentist_model, priors)
    lifted_model = lifted_module()
    with pyro.plate("customers", n):
        y_pred = lifted_model(X).squeeze(1)
        pyro.sample("obs", Bernoulli(y_pred, validate_args=True), obs=y)
        return y_pred


def predictive_model(x, y):
    pyro.sample("prediction", Delta(bayes_logistic(x, y)))


df_cust: pd.DataFrame = pd.read_csv("./data/customer-data.csv").\
    drop("id", axis=1).\
    dropna()

# use dictionary to construct feature transformation
transformation: Dict[str, Dict[str, Any]] = {}

df_cust["postal_code"] = df_cust["postal_code"].astype("object")
for col in df_cust.select_dtypes(include=["object"]):
    df_cust.loc[:, col] = df_cust.loc[:, col].astype("category")
    transformation.update({col: {v: k for (k, v) in enumerate(
                                        df_cust.loc[:, col].cat.categories)}})

for col in df_cust.select_dtypes(include=["bool"]):
    df_cust.loc[:, col] = df_cust.loc[:, col].astype(int)

# encode education and income levels manually to reflect increasing.
transformation.update(
    {
        "education": {
            "none": 0,
            "high school": 1,
            "university": 2
        },
        "income": {
            "poverty": 0,
            "working class": 1,
            "middle class": 2,
            "upper class": 3
        }
    }
)

df_cust = df_cust.replace(transformation)
df_features = df_cust.loc[:, df_cust.columns.difference(["outcome"])]
df_target = df_cust["outcome"]

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=20)

rf.fit(df_features, df_target)
top_k_feats = rf.feature_importances_.argsort()[-10:]
X, y = df_cust.loc[:, df_features.columns[top_k_feats]], df_target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, train_size=.8, shuffle=True, stratify=df_target)

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

k = X_tensor.shape[1] - 1
frequentist_model = TorchLogisticRegression(k)
q = AutoDiagonalNormal(bayes_logistic)
svi = SVI(bayes_logistic, q,
          Adam({"lr": 1e-2}),
          loss=Trace_ELBO(),
          num_samples=1000)

pyro.clear_param_store()
for i in range(3000):
    elbo = svi.step(X_tensor, y_tensor)
    if not i % 100:
        print(elbo / X_tensor.size(0))

svi_meanfield_posterior = svi.run(X_tensor, y_tensor)
new_sites = [f"parameter_{i}" for i in X_train.columns]
sites = ["bayes_logistic$$$linear.weight", "bayes_logistic$$$linear.bias"]


old_svi_samples = \
    {site: EmpiricalMarginal(svi_meanfield_posterior, sites=site)
     .enumerate_support().detach().cpu() for site in sites}
svi_samples = seperate_posteriors(old_svi_samples, new_sites)

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12, 10),
                        sharex=True,
                        sharey=True)
fig.suptitle("Marginal Posterior Distributions", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    try:
        site = new_sites[i]
        sns.distplot(svi_samples[site], ax=ax, label="Posterior Distribution")
        ax.set_title(site, fontsize=12)
    except IndexError:
        break
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.savefig("posterior_dists.png")


# sense check with standard package implementation of logistic regression
sklearn_model = LogisticRegression(solver="lbfgs")
sklearn_model.fit(X_train_scaled, y_train)

trace_pred = TracePredictive(predictive_model,
                             svi_meanfield_posterior,
                             num_samples=1000)
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_test.values, dtype=torch.float32)
posterior_predictive = trace_pred.run(X_tensor, None)
sites = ["prediction", "obs"]
posterior_predictive_samples = \
    {site: EmpiricalMarginal(posterior_predictive, sites=site)
     .enumerate_support().detach().cpu() for site in sites}

subset = posterior_predictive_samples["prediction"][:, 10:20]
y_pred_sklearn = sklearn_model.predict(X_test_scaled)
subset_sklearn = sklearn_model.predict_proba(X_test_scaled)[10:20, 1]

fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(12, 10),
                        sharex=True,
                        sharey=True)
fig.suptitle("Posterior Predictive Distributions", fontsize=16)
for i, ax in enumerate(axs.reshape(-1)):
    try:
        sns.distplot(subset[:, i], ax=ax, label="Posterior Predictive")
        ax.axvline(subset_sklearn[i], 0, ymax=.5, color='b', label="Delta")
        ax.set_title(f"unseen_sample_{i}", fontsize=12)
    except IndexError:
        break
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.savefig("posterior_predictive_dists.png")
