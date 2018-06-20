#! /usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
testing

brush up hypothesis testing and plotting.
"""
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, kstest, chisquare
from scipy.stats import chi2_contingency
from scipy.stats import norm, lognorm
import matplotlib as mpl
import seaborn as sns

mpl.rcParams['figure.dpi'] = 150
mpl.style.use('ggplot')

import warnings
warnings.filterwarnings("ignore")

alpha = .05
np.random.seed(0)

# one sample t test (sample mean vs popluation mean)
a = np.random.normal(30, 4, 44)
mean = 30
t, p = ttest_1samp(a, mean)
print(t, p)
if p < alpha:
    print('H0 rejected that a and popluation mean are different.')


# independent samples t test (i.e. two sample t test)
# H0 is that of same popluation means where two samples drawn from.
a = np.random.normal(2, .5, 1000)
b = np.random.normal(15, 2, 50)
t, p = ttest_ind(a, b, equal_var=True)
print(t, p)
if p < alpha:
    print('H0 rejected that a, b popluations have significant difference.')

# Kolmogorov-Smirnov test goodness of fit
# H0 is that a, b come from same distribution family.
d, p = kstest(a, 'norm', norm.fit(a))
print(d, p)
d, p = kstest(np.exp(a), 'lognorm', lognorm.fit(np.exp(a)))
print(d, p)
sns.distplot(a, kde=True)
sns.distplot(np.exp(a), kde=True)
# plt.show()

# Chi-Squared test goodness of fit
a = np.random.choice(4, 1000, p=[.05, .05, .05, .85])
b = np.random.randint(0, 4, 1000)
chisq, p = chisquare(np.bincount(a))
print(chisq, p)
if p < alpha:
    print('H0 rejected that a is not consistent to expected frequencies.')

# Chi-Squared test of independence
df = pd.DataFrame({'favor': [138, 64],
                   'indifferent': [83, 67],
                   'opposed': [64, 84]}, index=['democrat', 'republican'])
chisq, p, dof, ex = chi2_contingency(df)
print(chisq, p)
if p < alpha:
    print('H0 that there is independence between a & b. is rejected.')
