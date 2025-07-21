from scipy import stats
from numpy import sqrt

import datetime
import math

def blackscholes(stock_price,strike_price, IV,RFR, timeTomaturity):
    d1=math.log(stock_price/strike_price)+(RFR+IV*IV*0.5)*timeTomaturity/ (IV * sqrt(timeTomaturity))
    d2= d1- IV * math.sqrt(timeTomaturity)

    Nd1=stats.norm.cdf(d1)
    Nd2=stats.norm.cdf(d2)

    NPrimed1 = math.exp(-d1*d1/2)/math.sqrt(2*math.pi)

    price = stock_price * Nd1 - strike_price*Nd2*(math.exp(-RFR*timeTomaturity))

    delta =Nd1
    gamma=(NPrimed1)/(stock_price*IV*math.sqrt(timeTomaturity))
    vega=stock_price*sqrt(timeTomaturity)*NPrimed1

    return price,delta,gamma,vega

days_to_expiry =(datetime.date(2025,3,27) - datetime.date(2025,3,17)).days/365

price,delta,gamma,vega=blackscholes(19564.5,19500,0.0865,0.1,days_to_expiry)

print(round(price,2),round(delta,2),round(gamma,2),round(vega,2))