import numpy as np
from scipy import stats

class OptionPricing:

    def __init__(self, S0, K, r, sigma, T):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def bs_call_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + (self.sigma**2)/2) * self.T) / self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)
        self.bs_c = self.S0 * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        return self.bs_c

    def bs_put_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + (self.sigma**2)/2) * self.T) / self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)
        self.bs_p = self.K * stats.norm.cdf(-d2) - self.S0 * np.exp(-self.r * self.T) * stats.norm.cdf(-d1)
        return self.bs_p
    
    def mc_call_price(self, n_sims=1000, steps=100):
        dt = self.T / steps
        payoff_sum = 0
        for i in range(n_sims):
            S_T = self.S0
            for j in range(steps):
                phi = np.random.normal(0,1)
                S_T *= np.exp((self.r - (self.sigma**2 / 2))*dt + self.sigma * phi * np.sqrt(dt))
            payoff_sum += max(S_T-self.K, 0)
        self.mc_c = np.exp(-self.r * self.T) * (payoff_sum / n_sims)
        return self.mc