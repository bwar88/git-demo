# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:06:18 2016

@author: Ben
"""

import pandas as pd
import numpy as np
import scipy.optimize as ss
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.tsa.filters.filtertools as sm
import sklearn

df=pd.read_csv(r'C:\Users\Ben\Desktop\FR weekly csv.csv')

# Define Adstock Function
def adstock(x, rate, y):
    y['adstock'] = sm.recursive_filter(x,rate)
    return y

    
adstock(df['TV'],0.5,df)

model=smf.OLS(df['Offline_Conversions'],df['adstock']).fit().resid
rest=np.sqrt(model.resid**2)


# Run Optimization

def opt(df,x0):
    
    model=smf.OLS(df['Offline_Conversions'],adstock(df['TV'],x0,df).fit().resid    
    res=ss.minimize(model, x0)
    return res
                   
bw=opt(df,9999999)
    
    
test1=55
test2='be'

print ('hi %s and  %d sdad %d' %(test2,test1, test1) )
    