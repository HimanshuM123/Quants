import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import  scipy.optimize as optimization


NUM_TRADING_DAYS=252
#we will generate random w (Different portfolio)
NUM_PORTFOLIOS = 10000
stocks =['AAPL', 'WMT', 'TSLA', 'GE','AMZN','DB']

start_date ='2012-01-01'
end_date='2017-01-01'


def download_data():
    stock_data={}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date,end=end_date)['Close']

    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10,5))
    plt.show()

def calculate_return(data):
    log_return = np.log(data/data.shift(1))
    return log_return[1:]

def show_statistics(returns):
    print(returns.mean() * NUM_TRADING_DAYS)
    print(returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns , weights):
    portfolio_return = np.sum(returns.mean()*weights) * NUM_TRADING_DAYS
    portfolio_variance = np.dot( weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights))

    portfolio_volatility = np.sqrt(portfolio_variance)

    print("Expected portfolio mean return mean()", portfolio_return)
    print("Expected portfolio volatility return standard deviation()", portfolio_volatility)

def show_portfolio(returns , volatilities):
    plt.figure(figsize=(10,6))
    plt.scatter( volatilities,returns,c=returns/volatilities ,  marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharp Ratio')
    plt.show()

def show_optimal_portfolio(opts, rets , portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10,6))
    plt.scatter( portfolio_vols,portfolio_rets,c=portfolio_rets/portfolio_vols ,  marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharp Ratio')
    plt.plot(statistics(opts['x'],rets)[1], statistics(opts['x'],rets)[0] , 'g*', markersize=20)
    plt.show()



def generate_portfolios(returns):
    portfolio_means=[]
    portfolio_risks=[]
    portfolio_weights=[]

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /=np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean()*w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot( w.T, np.dot(returns.cov()*NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights) , np.array(portfolio_means) , np.array(portfolio_risks)


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean()* weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot( weights.T, np.dot(returns.cov()*NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

#The maximum of f(x) is minimum of -f(x)
def min_function_sharp(weights, returns):
    return -statistics(weights, returns)[2]

# what are the constraints? sum of weights =1
# f(x) =0 this has to be minimized
# sum =1 ie sum-1=0
def optimize_portfolio(weights, returns):
    constarints ={'type':'eq' , 'fun': lambda x : np.sum(x)-1}
    bounds = tuple((0,1) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharp, x0=weights[0] , args = returns, method ='SLSQP' , bounds=bounds , constraints=constarints)

def print_optimal_portfolio(optimum, returns):
    print('Optimal portfolio ', optimum['x'].round(3))
    print('Expected Return , volatility , sharp ratio', statistics(optimum['x'].round(3), returns))


if __name__ == '__main__':
    # print(download_data())
    data_set = download_data()
    show_data(data_set)
    log_daily_return = calculate_return(data_set)
    # show_statistics(log_daily_return)

    pweights, means , risks = generate_portfolios(log_daily_return)
    show_portfolio(means ,  risks)
    optimum= optimize_portfolio(pweights,log_daily_return)
    print_optimal_portfolio(optimum , log_daily_return)
    show_optimal_portfolio(optimum, log_daily_return,means,risks)






