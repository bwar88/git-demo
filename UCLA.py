# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:49:23 2016

@author: Ben
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

# read the data in
df = pd.read_csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# take a look at the dataset
print (df.head())

# rename the 'rank' column because there is also a DataFrame method called 'rank'
df.columns = ["admit", "gre", "gpa", "prestige"]
print (df.columns)

# summarize the data
print (df.describe())


# take a look at the standard deviation of each column
print (df.std())


# frequency table cutting presitge and whether or not someone was admitted
crosstb = pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])

# plot all of the columns
df.hist()
pl.show()

# dummify rank
dummy_ranks = pd.get_dummies(df['prestige'], prefix='prestige')
print (dummy_ranks.head())

# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = df[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_2':])
print (data.head())

# manually add the intercept
data['intercept'] = 1.0



#do the regression!!!!
train_cols = ['gre', 'gpa','prestige_2','prestige_3','prestige_4','intercept']
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

logit = sm.Logit(data['admit'], data[train_cols])

# fit the model
result = logit.fit()

# cool enough to deserve it's own gist
print (result.summary())

# look at the confidence interval of each coeffecient
print (result.conf_int())


# odds ratios only
print (np.exp(result.params))
#odds of being accepted decrease by 50% if school is tier 2

# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'params']
print (np.exp(conf))



# make predictions on the enumerated dataset
pre= result.predict(data[train_cols])





