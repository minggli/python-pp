#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import tensorflow_probability as tfp
from tensorflow_probability.distributions import Bernoulli, Normal

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

