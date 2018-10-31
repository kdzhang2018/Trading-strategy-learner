"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import util as ut
import matplotlib.pyplot as plt
import StrategyLearner as sl

def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)
    
    if verb: print "training"
    # set parameters for training the learner
    sym = "AAPL" #"SINE_FAST_NOISE" #"UNH" #"AAPL" #"ML4T-220"
    stdate =dt.datetime(2008,1,1)
    enddate =dt.datetime(2009,12,31) 
    #enddate =dt.datetime(2008,3,31)
    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, \
        ed = enddate, sv = 1000000) 
    
    if verb: print "in-sample testing"
    df_trades = learner.testPolicy(symbol = sym, sd = stdate, \
        ed = enddate, sv = 1000000)
    
#     if verb: print "out-of-sample testing" 
    stdate =dt.datetime(2010,1,1)
    enddate =dt.datetime(2011,12,31)
#     stdate =dt.datetime(2008,4,1)
#     enddate =dt.datetime(2008,6,30)
    
    # test the learner
#     df_trades = learner.testPolicy(symbol = sym, sd = stdate, \
#         ed = enddate, sv = 1000000)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    #if verb: print prices
    
    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 500, 0, -500
#     if isinstance(df_trades, pd.DataFrame) == False:
#         print "Returned result is not a DataFrame"
#     if prices.shape != df_trades.shape:
#         print "Returned result is not the right shape"
#     tradecheck = abs(df_trades.cumsum()).values
#     tradecheck[tradecheck<=500] = 0
#     tradecheck[tradecheck>0] = 1
#     if tradecheck.sum(axis=0) > 0:
#         print "Returned result violoates holding restrictions (more than 500 shares)"

    #if verb: print df_trades

if __name__=="__main__":
    test_code(verb = True)
