import QuantLib as ql
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from functions.probabilities import Heston_pdf, Q1, Q2


data_set = pd.read_csv('training set.csv').iloc[:,1:] 

N_sample = data_set.shape[0]

S0 = 100.0
limit_max = 10000      # right limit in the integration 
def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma*rho*u*1j
    d = np.sqrt( xi**2 + sigma**2 * (u**2 + 1j*u) )
    g1 = (xi+d)/(xi-d)
    g2 = 1/g1
    cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi-d)*t - 2*np.log( (1-g2*np.exp(-d*t))/(1-g2) ))\
              + (v0/sigma**2)*(xi-d) * (1-np.exp(-d*t))/(1-g2*np.exp(-d*t)) )
    return cf

heston_price_check = np.zeros(N_sample)
#condition = np.zeros(N_sample)

for i in range(N_sample):

	T = data_set.loc[i,'mat_data_range']/360
	K = S0*data_set.loc[i,'strike']
	#dividend_rate =  round(data_set.loc[i,'divident'],3)
	r = data_set.loc[i,'interest']
	v0 = data_set.loc[i,'v_0']
	kappa = data_set.loc[i,'kappa_']
	sigma = data_set.loc[i,'theta_']
	theta = data_set.loc[i,'ita_']
	rho = data_set.loc[i,'rho_']
	k = np.log(K/S0)
	cf_H_b_good = partial(cf_Heston_good, t=T, v0=v0, mu=r, theta=theta, sigma=sigma, kappa=kappa )

	limit_max = 1000      # right limit in the integration                
	call = S0 * Q1(k, cf_H_b_good, limit_max) - K * np.exp(-r*T) * Q2(k, cf_H_b_good, limit_max)
	heston_price_check[i] = call
	if i%100==0:
		print(i)
	
	

data_set['checkPrice'] = heston_price_check

data_set.to_csv('checked_price.csv')

# diff = data_set[abs(data_set['heston_price'] - data_set['checkPrice'])>0.5]
# diff.to_csv('diff.csv')

# test_set = data_set[abs(data_set['heston_price'] - data_set['checkPrice'])<0.1]
# test_set = test_set.iloc[:100000,:10]
# test_set.to_csv('test set.csv')

# neg = data_set[data_set['heston_price']<0]
# neg.to_csv('neg.csv')