# -*- coding: utf-8 -*-
"""
Hochschule Albstadt-Sigmaringen 
Praktikum Betriebssicherheit SS2024
Aufgabenblatt 1

@author: 
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter 



### (a) 
#load data
data = pd.read_excel('Aufgabe1_data.xlsx')
data_array = data.loc[:,0].to_numpy()

#visualize data / show exponential ditribution
data_array_sort = data_array.copy()
data_array_sort.sort()
plt.figure()   
plt.title('Data Visualization')
plt.plot(data_array)
plt.plot(data_array_sort)

n = len (data_array)
x_mean = data_array.sum()  / n 

#Kaplan-Maier
plt.figure()  
kmf = KaplanMeierFitter()
kmf.fit(data_array)
kmf.plot()


### (b)
lambda_hat = 1 / x_mean
R_t = lambda t: np.exp(-lambda_hat * t)
t = np.linspace(0,2,101)
plt.figure()   
plt.title('Zuverlässigkeitsfunktion')
plt.plot(t, R_t(t))



### (c) 
#load data
data_cens = pd.read_excel('Aufgabe1_data_cens.xlsx')
data_array_cens = data_cens.loc[:,0].to_numpy()

#visualize data / show exponential ditribution
data_array_sort_cens = data_array_cens.copy()
data_array_sort_cens.sort()
plt.figure()   
plt.title('Data Visualization')
plt.plot(data_array_cens)
plt.plot(data_array_sort_cens)

#Kaplan-Maier
plt.figure()  
kmf_cens = KaplanMeierFitter()
kmf_cens.fit(data_array_cens)
kmf_cens.plot()


### (d)
n_cens = len (data_array_cens)
data_array_cens_r = np.array( [x for x in data_array_cens if x != 0.25] )
r = len(data_array_cens_r)
T = 0.25

lambda_hat_cens = r / ( data_array_cens_r.sum()  + (n_cens - r)*T )

R_t_cens = lambda t: np.exp(-lambda_hat_cens * t)
t_cens = np.linspace(0,0.25,101)
plt.figure()   
plt.title('Zuverlässigkeitsfunktion')
plt.plot(t_cens, R_t_cens(t_cens))
