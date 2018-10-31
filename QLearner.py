"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""
import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.Q = np.zeros((num_states, num_actions))
        
        if self.dyna != 0:
            self.experience = np.empty((10000, 4))
            self.count = 0
        
        self.s = 0
        self.a = 0

    def author(self):
        return 'kzhang346' # replace tb34 with your Georgia Tech username.
        
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s, :])
        self.s = s
        self.a = action
        #if self.verbose: print "Set state: s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #if self.verbose: 
        #    print "s =", self.s ,"a =",self.a, "s' =", s_prime, "r =", r 
        #    print "old Q =",self.Q[self.s, self.a]
        
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]))
        
        if self.verbose: 
            #print "new Q =",self.Q[self.s, self.a]
            if r == 1: 
                #print self.Q
                print "count= ", self.count
        
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions-1)
            #if self.verbose: print "random action =", action
        else:
            action = np.argmax(self.Q[s_prime, :])
            #if self.verbose: print "non-random action =",  action
    
        if self.dyna != 0:
            self.experience[self.count, :] = (self.s, self.a, s_prime, r)
            self.count += 1
            #if self.verbose: print self.count
            for i in range(0, self.dyna):
                dyna_s, dyna_a, dyna_s_prime, dyna_r = self.experience[np.random.randint(self.count), :]
                self.Q[dyna_s, dyna_a] = (1 - self.alpha) * self.Q[dyna_s, dyna_a] + self.alpha * (dyna_r + self.gamma * np.max(self.Q[dyna_s_prime, :]))
        
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        #if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
