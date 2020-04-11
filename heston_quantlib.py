import QuantLib as ql
import time




# option data
t = time.time()
calculation_date = ql.Date(31, 3, 2020)
#maturity_date = ql.Date(15, 1, 2016)calculation_date.serialNumber().
maturity_date = ql.Date(calculation_date.serialNumber() + 336)
spot_price = 100
strike_price = 100*1.53825065
volatility = 0.043855505# the historical vols for a year
dividend_rate =  0.016356544
option_type = ql.Option.Call

risk_free_rate = 0.01697
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()


ql.Settings.instance().evaluationDate = calculation_date

# construct the European Option
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
exercise = ql.EuropeanExercise(maturity_date)
european_option = ql.VanillaOption(payoff, exercise)


v0 = 0.0218 # spot variance
kappa = 1.6836
theta = 0.0221
sigma = 0.1709
rho = -0.8145


spot_handle = ql.QuoteHandle(
    ql.SimpleQuote(spot_price)
)
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
                                  theta,
                                  sigma,
                                  rho)

engine = ql.AnalyticHestonEngine(ql.HestonModel(heston_process),0.00001, 10000)
european_option.setPricingEngine(engine)
h_price = european_option.NPV()
#gamma = european_option.delta()
print("The Heston model price is",h_price)
#print("The gamma of Heston model price is",gamma)
#print('takes {} seconds'.format(time.time() - t))
print(maturity_date)
