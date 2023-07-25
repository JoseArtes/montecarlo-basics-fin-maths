import numpy as np
from scipy import stats

class OptionPricing:
    """
    A class to calculate the price of European call & put options via two methods:
    1. Black-Scholes.
    2. Monte-Carlo.

    Attributes
    ----------
    S : float
        Stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Implied stock volatility.
    T : float
        Option maturity.
    
    Methods
    -------
    set_S():
        Stock setter.

    set_K():
        Strike price setter.

    set_r():
        Risk-free rate setter.

    set_sigma():
        Implied volatility setter.

    set_T():
        Option maturity setter.

    bs_call_price():
        Calculates call option price via Black-Scholes.
    
    bs_put_price():
        Calculates put option price via Black-Scholes.

    mc_call_price():
        Calculates call option price via Monte Carlo.

    mc_put_price():
        Calculates put option price via Monte Carlo.
    """

    def __init__(self, S, K, r, sigma, T):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def set_S(self, S):
        """Stock price setter.

        Args:
            S (float): Stock price.
        """
        self.S = S

    def set_K(self, K):
        """Strike price setter.

        Args:
            K (float): Strike price.
        """
        self.K = K

    def set_r(self, r):
        """Risk-free rate setter.

        Args:
            r (float): Risk-free rate.
        """ 
        self.r = r

    def set_sigma(self, sigma):
        """Implied volatility setter.

        Args:
            sigma (float): Implied volatility.
        """
        self.sigma = sigma

    def set_T(self, T):
        """Option maturity setter.

        Args:
            T (float): Option maturity.
        """
        self.T = T

    def bs_call_price(self):
        """Calculates call option price via Black-Scholes.

        Returns:
            bs_c (float): Call option price.
        """
        d1 = (np.log(self.S / self.K) + (self.r + (self.sigma**2)/2) * self.T) / self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)
        self.bs_c = self.S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        return self.bs_c

    def bs_put_price(self):
        """Calculates put option price via Black-Scholes.

        Returns:
            bs_p (float): Put option price.
        """
        d1 = (np.log(self.S / self.K) + (self.r + (self.sigma**2)/2) * self.T) / self.sigma * np.sqrt(self.T)
        d2 = d1 - self.sigma * np.sqrt(self.T)
        self.bs_p = self.K * stats.norm.cdf(-d2) - self.S * np.exp(-self.r * self.T) * stats.norm.cdf(-d1)
        return self.bs_p
    
    def mc_call_price(self, n_sims=1000, steps=100):
        """Calculates call option price via Black-Scholes.

        Args:
            n_sims (int, optional): Number of simulations. Defaults to 1000.
            steps (int, optional): Number of time steps for each simulation. Defaults to 100.

        Returns:
            mc_c: Call option price.
        """
        dt = self.T / steps
        payoff_sum = 0
        for i in range(n_sims):
            S_T = self.S
            for j in range(steps):
                phi = np.random.normal(0,1)
                S_T *= np.exp((self.r - (self.sigma**2 / 2))*dt + self.sigma * phi * np.sqrt(dt))
            payoff_sum += max(S_T-self.K, 0)
        self.mc_c = np.exp(-self.r * self.T) * (payoff_sum / n_sims)
        return self.mc_c
    
    def mc_put_price(self, n_sims=1000, steps=100):
        """Calculates put option price via Monte Carlo.

        Args:
            n_sims (int, optional): Number of simulations. Defaults to 1000.
            steps (int, optional): Number of time steps for each simulation. Defaults to 100.

        Returns:
            mc_p: Put option price.
        """
        dt = self.T / steps
        payoff_sum = 0
        for i in range(n_sims):
            S_T = self.S
            for j in range(steps):
                phi = np.random.normal(0,1)
                S_T *= np.exp((self.r - (self.sigma**2 / 2))*dt + self.sigma * phi * np.sqrt(dt))
            payoff_sum += max(self.K-S_T, 0)
        self.mc_p = np.exp(-self.r * self.T) * (payoff_sum / n_sims)
        return self.mc_p