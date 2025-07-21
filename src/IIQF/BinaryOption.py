from scipy import stats
from numpy import sqrt

import datetime
import math

def binary_blackscholes(stock_price,strike_price, IV,RFR, timeTomaturity):
    d1=math.log(stock_price/strike_price)+(RFR+IV*IV*0.5)*timeTomaturity/ (IV * sqrt(timeTomaturity))
    d2= d1- IV * math.sqrt(timeTomaturity)

    Nd1=stats.norm.cdf(d1)
    Nd2=stats.norm.cdf(d2)



    return math.exp(-RFR*timeTomaturity)*stats.norm.cdf(d2)

days_to_expiry =(datetime.date(2025,3,27) - datetime.date(2025,3,17)).days/365

price = binary_blackscholes(19564.5,19500,0.0865,0.1,days_to_expiry)

print(round(100*price,2))