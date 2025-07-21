#solving price of option using Binomial Model
import numpy as np

def binomial_tree_option_values(stockPrice, strikePrice, riskFreRate, timeToMaturity,sigma, n , option_type="call"):
    delta_T = timeToMaturity/n
    u=1.05
    d = 1/u
    p = (np.exp(riskFreRate*delta_T)-d)/(u-d)
    option_values = np.zeros((n+1, n+1))

    for j in range (n+1):
        option_values[n,j]=max(0,(strikePrice*u**j*d**(n-j))-stockPrice) if option_type=='call' else max(0,stockPrice-(stockPrice*u**j*d**(n-j)))

    for i in range(n-1,-1,-1):
        for j in range(i+1):
            option_values[i,j]=np.round((p*option_values[i+1,j+1]+(1-p)*option_values[i+1,j]) * (np.exp(-riskFreRate*delta_T)),2)
    return option_values



stock_price = 100
stike_price =100
RFR=0.05
timeToMat=0.75
sigma=0.0976
N=3

option_tree=binomial_tree_option_values(stock_price,stike_price,RFR,timeToMat,sigma,N,option_type='call')

print(option_tree)