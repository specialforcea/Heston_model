import QuantLib as ql
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



N_sample = 100000


#constants

calculation_date = ql.Date(31, 3, 2020)
serial_today = calculation_date.serialNumber()#print(calculation_date.serialNumber())
spot_price = 100.0
option_type = ql.Option.Call
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
ql.Settings.instance().evaluationDate = calculation_date
spot_handle = ql.QuoteHandle(
      ql.SimpleQuote(spot_price)
  )

#

#training data
mat_data_range = np.random.randint(30,70,size=N_sample)
strike = np.random.uniform(0.5,1.5,size=N_sample)
interest = np.random.uniform(0.015,0.025,size=N_sample)
divident = np.random.uniform(0,0.05,size=N_sample)
kappa_ = np.random.uniform(1.5,2.5,size=N_sample)
rho_ = np.random.uniform(-0.8,-0.6,size=N_sample)
theta_ = np.random.uniform(0.1,0.2,size=N_sample)
cum_ita = np.random.uniform(0.,1.0,size=N_sample)
ita_ = -np.log(np.exp(-0.02) - cum_ita*(np.exp(-0.02)-np.exp(-0.1)))
cum_v_0 = np.random.uniform(0.,1.0,size=N_sample)
v_0 = -np.log(np.exp(-0.02) - cum_v_0*(np.exp(-0.02)-np.exp(-0.1)))
heston_price = np.zeros(N_sample)

# print(type(mat_data_range[10]),type(serial_today),type(mat_data_range[10] + serial_today))
# print(ql.Date(serial_today + int(mat_data_range[10])))



t = time.time()
for i in range(N_sample):
  
  maturity_date = ql.Date(serial_today + int(mat_data_range[i]))
  strike_price = spot_price*strike[i]
  dividend_rate =  divident[i]
  risk_free_rate = interest[i]
  v0 = v_0[i]
  kappa = kappa_[i]
  theta = theta_[i]
  ita = ita_[i] 
  rho = rho_[i]


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
  engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),10**-10, 100000)
  european_option.setPricingEngine(engine)
  heston_price[i] = european_option.NPV()

#print("The gamma of Heston model price is",gamma)
print('takes {} seconds'.format(time.time() - t))

#make and save dataframe
mat_data_range = mat_data_range.reshape(-1,1)
strike = strike.reshape(-1,1)
interest = interest.reshape(-1,1)
divident = divident.reshape(-1,1)
kappa_ = kappa_.reshape(-1,1)
rho_ = rho_.reshape(-1,1)
theta_ = theta_.reshape(-1,1)
ita_ = ita_.reshape(-1,1)
v_0 = v_0.reshape(-1,1)
heston_price = heston_price.reshape(-1,1)


training_set = np.hstack((mat_data_range,strike,interest,divident,
                          kappa_,rho_,theta_,v_0,ita_,heston_price))
print(training_set.shape,training_set[0])
train_set_df = pd.DataFrame(training_set,columns=['mat_data_range','strike','interest','divident','kappa_','rho_','theta_','v_0','ita_','heston_price'])

train_set_df.to_csv('test set.csv')
# plt.scatter(strike,heston_price)
# plt.show()