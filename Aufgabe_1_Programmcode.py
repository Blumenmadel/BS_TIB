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
plt.title('Data Visualization 1')
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
plt.title('Zuverlässigkeitsfunktion 1')
plt.plot(t, R_t(t))



### (c) 
#load data
data_cens = pd.read_excel('Aufgabe1_data_cens.xlsx')
data_array_cens = data_cens.loc[:,0].to_numpy()

#visualize data / show exponential ditribution
data_array_sort_cens = data_array_cens.copy()
data_array_sort_cens.sort()
plt.figure()   
plt.title('Data Visualization 2')
plt.plot(data_array_cens)
plt.plot(data_array_sort_cens)

n_cens = len (data_array_cens)
x_mean_cens = data_array_cens.sum()  / n_cens


#Kaplan-Maier
plt.figure()  
kmf_cens = KaplanMeierFitter()
kmf_cens.fit(data_array_cens)
kmf_cens.plot()


### (d)
lambda_hat_cens = 1 / x_mean_cens
R_t_cens = lambda t: np.exp(-lambda_hat_cens * t)
t_cens = np.linspace(0,0.25,101)
plt.figure()   
plt.title('Zuverlässigkeitsfunktion 2')
plt.plot(t_cens, R_t_cens(t_cens))
