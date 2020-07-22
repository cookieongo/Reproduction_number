#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
from numpy import exp
import matplotlib.pyplot as plt
import math
from datetime import date
import seaborn as sns
import statistics as st
from scipy.integrate import odeint


# In[78]:


df1=pd.read_csv('StatewiseTestingDetails.csv')
df2=pd.read_csv('data2.csv')
df2['TotalSamples']=df2['TotalSamples'].fillna(0)
s=df2.Population-df2.Cured-df2.Deaths-df2.Confirmed-df2.TotalSamples
df2['Sus']=s
dt1=df2[df2.Days<36]
plt.style.use('seaborn-whitegrid')
dt1.plot(x='Days',y=['Confirmed','Cured','Deaths'])
dt1.plot(x='Days',y='Sus')
plt.title('Number of cases per day')
df3=pd.read_csv('datasets_557629_1357144_population_india_census2011.csv')


# In[79]:


from scipy.stats import expon
growth=[0 for x in range(len(df2))]
for i in range(2,len(df2)):
  growth[i]=(df2.Confirmed[i]-df2.Confirmed[i-1])/df2.Confirmed[i-1]

df2['growth']=growth

df2['new']=250*exp(df2.growth*df2.Days)
R0=1/st.mean(exp(-df2.growth*df2.Days))

print('Reprodction number using exponential', R0)


# In[85]:


beta=[0 for x in range(len(df2))]
R0=[0 for x in range(len(df2))]
beta[0]=0
gamma=1/18
for i in range(2,len(df2)):
  beta[i]=(df2.Confirmed[i]-df2.Confirmed[i-1])/df2.Sus[i]
  R0[i]=beta[i]/gamma

plt.title('Transmission Rate')
plt.plot(df2.Days,beta)

df2.Confirmed[139]/(140*df2.Population[1])


# In[86]:


def model(x,t,beta,gamma):
#x= sus, infec, reco
  dsdt=-beta*x[0]*x[1]/df2.Population[1]
  didt=(beta*x[0]*x[1]/df2.Population[1])-(gamma*x[1])
  drdt=gamma*x[1]

  return dsdt,didt,drdt


# In[89]:


R0= np.linspace(1.2, 3.0, 6)
t=range(1,440)
x=df2.Sus[139], df2.Confirmed[139],df2.Cured[139]
fig, ax = plt.subplots()
for r in R0:
  z=odeint(model,x,t,args=(r,gamma))
  ax.plot(z[:,[2]], label=r)
  
plt.xlabel('Number of days')
plt.ylabel('Number of people recovered (1e8)')
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)


# In[90]:


R0= np.linspace(1.2, 3.0, 6)
t=range(1,440)
x=df2.Sus[139], df2.Confirmed[139],df2.Cured[139]
fig, ax = plt.subplots()
for r in R0:
  z=odeint(model,x,t,args=(r,gamma))
  ax.plot(z[:,[1]], label=r)
  
plt.xlabel('Number of days')
plt.ylabel('Infected People')
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)






