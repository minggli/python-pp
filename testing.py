#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
testing

brush up hypothesis testing and plotting.
"""

import numpy as np
from scipy.stats import ttest_ind, ttest_1samp

alpha = .05

# one sample t test (sample mean vs popluation mean)
a = np.random.normal(30, 5, 44)
mean = 30
t, p = ttest_1samp(a, mean)
print(t, p)
if p < alpha:
    print('H0 rejected that a and popluation mean are different.')


# independent samples t test (i.e. two sample t test)
# H0 is that of same popluation means where two samples drawn from.
a = np.random.normal(10, 4, 30)
b = np.random.normal(15, 2, 50)
t, p = ttest_ind(a, b, equal_var=True)
print(t, p)
if p < alpha:
    print('H0 rejected that a, b popluations have significant difference.')
