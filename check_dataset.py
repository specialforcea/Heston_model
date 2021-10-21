import QuantLib as ql
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data_set = pd.read_csv('training set.csv').iloc[:,1:] 

N_sample = data_set.shape[0]

calculation_date = ql.Date(31, 3, 2020)
serial_today = calculation_date.serialNumber()#print(calculation_date.serialNumber())
spot_price = 100.0
option_type = ql.Option.Call
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
ql.Settings.instance().evaluationDate = calculation_date
spot_handle = ql.QuoteHandle(
      ql.SimpleQuote(spot_price))

heston_price_check = np.zeros(N_sample)
condition = np.zeros(N_sample)

for i in range(N_sample):

	maturity_date = ql.Date(serial_today + int(data_set.loc[i,'mat_data_range']))
	strike_price = spot_price*data_set.loc[i,'strike']
	dividend_rate =  data_set.loc[i,'divident']
	risk_free_rate = data_set.loc[i,'interest']
	v0 = data_set.loc[i,'v_0']
	kappa = data_set.loc[i,'kappa_']
	theta = data_set.loc[i,'theta_']
	ita = data_set.loc[i,'ita_']
	rho = data_set.loc[i,'rho_']


	payoff = ql.PlainVanillaPayoff(option_type, strike_price)
	exercise = ql.EuropeanExercise(maturity_date)
	european_option = ql.VanillaOption(payoff, exercise)


	flat_ts = ql.YieldTermStructureHandle(
	  ql.FlatForward(calculation_date, risk_free_rate, day_count)
	)
	dividend_yield = ql.YieldTermStructureHandle(
	  ql.FlatForward(calculation_date, dividend_rate, day_count)
	)
	heston_process = ql.HestonProcess(flat_ts,
	                                dividend_yield,
	                                spot_handle,
	                                v0,
	                                kappa,
	                                ita,
	                                theta,
	                                rho)
	#print(v0,kappa,  ita,  theta,rho)
	engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),10**-8, 100000)
	european_option.setPricingEngine(engine)
	heston_price_check[i] = european_option.NPV()
	if i%200==0:
		print(i)
	if 2*kappa*ita>theta**2:
		condition[i] = 1
	else:
		condition[i] = 0

data_set['checkPrice'] = heston_price_check
data_set['condition'] = condition
#data_set.to_csv('checked_price.csv')

diff = data_set[abs(data_set['heston_price'] - data_set['checkPrice'])>0.0001]
diff.to_csv('diff.csv')

# test_set = data_set[abs(data_set['heston_price'] - data_set['checkPrice'])<0.1]
# test_set = test_set.iloc[:100000,:10]
# test_set.to_csv('test set.csv')

# neg = data_set[data_set['heston_price']<0]
# neg.to_csv('neg.csv')