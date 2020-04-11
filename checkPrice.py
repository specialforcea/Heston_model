import QuantLib as ql
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



calculation_date = ql.Date(31, 3, 2020)
serial_today = calculation_date.serialNumber()#print(calculation_date.serialNumber())
spot_price = 100.0
option_type = ql.Option.Call
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
ql.Settings.instance().evaluationDate = calculation_date
spot_handle = ql.QuoteHandle(
      ql.SimpleQuote(spot_price))





maturity_date = ql.Date(serial_today + int(338))
strike_price = spot_price*1.568260549
dividend_rate =  0.048266176
risk_free_rate = 0.015936227
v0 = 0.09078221
kappa = 1.962850649
theta = 0.544034019
ita = 0.013006305
rho = -0.748117391


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
engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.01, 1000)
european_option.setPricingEngine(engine)
print(european_option.NPV())

