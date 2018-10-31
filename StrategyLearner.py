"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import QLearner as ql
import random
import matplotlib.pyplot as plt

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.SHARES = 200
        self.MARKET_IMPACT_PENALTY = 50 * 0.0001
        self.COMMISSION_COST = 9.95  
        self.lookback = 21

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), sv = 10000): 
        # get states
        self.syms = [symbol]
        self.sd = sd
        self.ed = ed
        self.sv = sv
        states, daily_rets, prices, benchmark = self.getStates()
        if self.verbose:
            print "benchmark: ", benchmark
        
        # initiate a QLearner
        self.learner = ql.QLearner(num_states=1000, num_actions = 3, alpha = 0.2, \
            gamma = 0.9, rar = 0.98, radr = 0.999, dyna = 0, verbose=False)
        # train the QLearner
        epochs = 200
        prev_port_val = 0 # final portfolio value in previous epoch
        for epoch in range(1, epochs+1):
            port_val = self.sv
            # set the initial state
            action = self.learner.querysetstate(states.ix[0])
            holding = self.actionToHolding(action)
            if holding != 0:
                reward = -1 * (self.COMMISSION_COST + pd.np.abs(holding) * prices.ix[0].values[0] * self.MARKET_IMPACT_PENALTY)
            else: reward = 0
            prev_holding = holding # holding on the day before
            
            for day in range(1, states.shape[0]):
                reward += prev_holding * daily_rets.ix[day].values[0]
                port_val += reward
                action = self.learner.query(states.ix[day], reward)
                holding = self.actionToHolding(action)
                # for next day: deduct transaction costs   
                if holding != prev_holding:
                    reward = -1 * (self.COMMISSION_COST + pd.np.abs(holding - prev_holding) * prices.ix[day].values[0] * self.MARKET_IMPACT_PENALTY)
                else: reward = 0
                prev_holding = holding
                
            if self.verbose:
                print epoch, ": ", port_val
            if epoch >= 20 and prev_port_val == port_val: break
            prev_port_val = port_val

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), ed=dt.datetime(2010,1,1), sv = 10000):
        self.syms = [symbol]
        self.sd = sd
        self.ed = ed
        self.sv = sv
        # get states
        states, daily_rets, prices, benchmark = self.getStates()        
        if self.verbose:
            print "benchmark: ", benchmark
        
        prev_holding = 0
        reward = 0
        holdings = states.copy()
        for day in range(0, states.shape[0]):
            holdings.ix[day] = prev_holding
            action = self.learner.querysetstate(states.ix[day])
            holding = self.actionToHolding(action)
            prev_holding = holding
        
        trades = holdings.copy()
        trades.values[:-1] = holdings.values[1:] - holdings.values[:-1]
        trades.values[-1] = holding - prev_holding # values from the last iteration
        
        if self.verbose:
            df = pd.DataFrame(index = prices.index)
            df['prices'] = (prices - prices.mean()) / (prices.max() - prices.min())
            df['daily_rets'] = daily_rets / (daily_rets.max() - daily_rets.min())
            df['holdings'] = holdings / self.SHARES / 2
            df[['holdings', 'daily_rets']].plot(title = 'ML4T-220 holdings and daily_rets')
            plt.legend(loc='lower right')
            plt.show()
        
        return trades
        
    def actionToHolding(self, action):
        if action == 0: holding = 0 # Zero shares
        elif action == 1: holding = self.SHARES # LONG
        elif action == 2: holding = -self.SHARES # SHORT
        return holding
                
    def getStates(self):
        dates = pd.date_range(self.sd - dt.timedelta(days = self.lookback * 2), self.ed)
        prices_all = ut.get_data(self.syms, dates)  # automatically adds SPY
        prices = prices_all[self.syms]  # only portfolio symbols
        # bollinger band
        sma = pd.rolling_mean(prices, window = self.lookback)
        rolling_std = pd.rolling_std(prices, window = self.lookback)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        # bollinger band percentage
        bbp = (prices - bottom_band) / (top_band - bottom_band)
        # sma ratio
        sma = prices / sma
        # daily returns
        daily_rets = prices.copy()
        daily_rets.values[1:, :] = prices.values[1:, :] - prices.values[:-1, :]
        daily_rets.values[0, :] = np.nan
        # relative strength index
        up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
        down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
        up_gain = prices.copy()
        up_gain.ix[:, :] = 0
        up_gain.values[self.lookback:, :] = up_rets.values[self.lookback:, :] - up_rets.values[:-self.lookback, :]
        down_loss = prices.copy()
        down_loss.ix[:, :] = 0
        down_loss.values[self.lookback:, :] = down_rets.values[self.lookback:, :] - down_rets.values[:-self.lookback, :]
        rs = (up_gain / self.lookback) / (down_loss / self.lookback)
        rsi = 100 - (100 / (1 + rs))
        rsi.ix[:self.lookback, :] = np.nan
        rsi[rsi == np.inf] = 100 # when down_loss is 0, rsi is 100

        # discretize three indicators and calculate states       
        bbp_disc = pd.qcut(bbp, 10, labels=False)
        sma_disc = pd.qcut(sma, 10, labels=False)
        rsi_disc = pd.qcut(rsi, 10, labels=False)
        states = pd.DataFrame(100 * bbp_disc + 10 * sma_disc + rsi_disc, index = bbp.index, columns = self.syms)
        
        # remove data before the start date
        prices = prices.ix[self.sd:]
        daily_rets = daily_rets.ix[self.sd:]
        states = states.ix[self.sd:]
        # benchmark: Buy shares on the first trading day, Sell shares on the last day.
        benchmark = self.sv + (prices.ix[-1] - prices.ix[0]).values[0] * self.SHARES
        benchmark -= self.COMMISSION_COST * 2 + (prices.ix[0].values[0] + prices.ix[-1].values[0]) * self.SHARES * self.MARKET_IMPACT_PENALTY
        
        return states, daily_rets, prices, benchmark

if __name__=="__main__":
    pass
    
