import numpy as np

class OptionPricing:

    def __init__(self, S0, E,T,rf,sigma, iterations):
        self.S0=S0 #actual price of stock
        self.E=E  # strike price
        self.T=T
        self.rf=rf
        self.sigma = sigma
        self.iterations =iterations


    def call_option_simulation(self):
        # we have 2 columns ,first with 0's & 2nd will store payoffs
        # first column =0 , second column = max(0, S-E)
        option_data = np.zeros([self.iterations,2])
        # print(option_data)
        # Dimensions : 1 dimensional array with as many items as iterations
        rand = np.random.normal(0,1,[1, self.iterations])
        # print(rand)
        #Equation for S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T *(self.rf - 0.5 * self.sigma ** 2)
                            + self.sigma * np.sqrt(self.T) * rand)
        # print(stock_price)
        option_data[:,1] =  stock_price - self.E
        # average for monte-carlo simulation
        average = np.sum(np.amax(option_data, axis=1))/ float(self.iterations)
        # print(option_data)
        #have to use exp(-rT) discount factor ie present value of future cashflow
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        # we have 2 columns ,first with 0's & 2nd will store payoffs
        # first column =0 , second column = max(0, S-E)
        option_data = np.zeros([self.iterations,2])
        # print(option_data)
        # Dimentions : 1 dimentional array with as many items as iterations
        rand = np.random.normal(0,1,[1, self.iterations])
        # print(rand)
        #Equation for S(t) stock price at T
        stock_price = self.S0 * np.exp(self.T *(self.rf - 0.5 * self.sigma ** 2)
                            + self.sigma * np.sqrt(self.T) * rand)
        # (E-S)
        option_data[:,1] =   self.E - stock_price
        # average for monte-carlo simulation
        average = np.sum(np.amax(option_data, axis=1))/ float(self.iterations)
        # print(option_data)
        #have to use exp(-rT) discount factor ie present value of future cashflow
        return np.exp(-1.0 * self.rf * self.T) * average

if __name__ == '__main__':
    model = OptionPricing(100,100,1, 0.05,0.2,1000)
    print('Value of call option is $%.2f' % model.call_option_simulation())
    print('Value of put option is $%.2f' % model.put_option_simulation())
