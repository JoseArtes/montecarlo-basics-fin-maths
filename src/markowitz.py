import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Markowitz:
    """
    A class to calculate the Minimum Variance portfolio and Maximum Sharpe ratio via
    the Monte Carlo method and numerical approximation (SLSQP)

    Attributes
    ----------
    df_returns : pd.DataFrame
        Dataframe of stock returns.

    Methods
    -------
    _portfolio_returns():
        Computes the portfolio returns.

    _portfolio stdev():
        Computes the portfolio standard deviation.

    _sharpe ratio():
        Computes the Sharpe ratio of the portfolio.

    simulate_portfolios():
        Performs Monte Carlo simulation for portfolio creation.

    optimal_portfolio():
        Applies numerical optimisation via the SLSQP method to find out the optimal
        portfolio either based on minimum-variance or maximum Sharpe ratio.
    """

    def __init__(self, df_returns):
        self.df_returns = df_returns
        self.n_stocks = self.df_returns.shape[1]
        self.mu_p = self.df_returns.mean() * 252
        self.cov = self.df_returns.cov() * 252

    def _portfolio_returns(self, w):
        """Calculated portfolio returns.

        Args:
            w (np.array): Array of weights.

        Returns:
            float: Portfolio return.
        """
        return w @ self.mu_p

    def _portfolio_stdev(self, w):
        """Calculated portfolio standard deviation.

        Args:
            w (np.array): Array of weights.

        Returns:
            float: Portfolio return.
        """
        return np.sqrt(w @ self.cov @ w)

    def _sharpe_ratio(self, w):
        """Calculate portfolio Sharpe ratio.

        Args:
            w (np.array): Array of weights.

        Returns:
            float: Portfolio Sharpe ratio.
        """
        return -1 * self._portfolio_returns(w) / self._portfolio_stdev(w)

    def simulate_portfolios(self, n_sims=1000):
        """Simulate a user-specified number of portfolios.

        Args:
            n_sims (int, optional): Number of simulated portfolios. Defaults to 1000.

        Returns:
            pd.DataFrame: Dataframe containing the weights and metrics of interest (Portfolio Return, Stdev, Sharpe)
            for each simulation.
        """
        p_w = []
        p_returns = []
        p_stdev = []
        p_sharpe = []
        for n in range(n_sims):
            w = np.random.dirichlet([1] * self.n_stocks)
            p_w.append(w)
            p_returns.append(self._portfolio_returns(w))
            p_stdev.append(self._portfolio_stdev(w))
            p_sharpe.append(self._sharpe_ratio(w))

        w_df = pd.DataFrame(p_w, columns=[f"w{i}" for i in range(self.n_stocks)])
        self.df_sim = pd.DataFrame(
            {"mu_p": p_returns, "sigma_p": p_stdev, "sharpe": p_sharpe},
        )
        self.df_sim = pd.concat([w_df, self.df_sim], axis=1)
        self.df_sim["sharpe"] = -1 * self.df_sim["sharpe"]
        return self.df_sim

    def optimal_portfolio(self, obj="min-var"):
        """Function to calculate the optimal portfolio.

        Args:
            obj (str, optional): Objective function. Defaults to "min-var".
                - "min-var": minimum-variance portfolio.
                - "max-sharpe: maximum Sharpe ratio portfolio.

        Returns:
            pd.DataFrame: Dataframe containing the weights and metrics of interest (Portfolio Return, Stdev, Sharpe)
            for optimal portfolio.
        """
        # Initialise weights from Dirichlet(1,1,1,1) distribution
        w0 = np.random.dirichlet([1] * self.n_stocks)
        # Define constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        # Define bounds
        bounds = ((0, 1),) * self.n_stocks

        if obj == "min-var":
            obj_fun = self._portfolio_stdev
        elif obj == "max-sharpe":
            obj_fun = self._sharpe_ratio

        result = minimize(
            obj_fun, w0, method="SLSQP", constraints=constraints, bounds=bounds
        )

        # Extract weight results that reach the minimum of objective function
        self.opt_p_w = result.x
        w_dict = {f"w{i}": j for i, j in enumerate(self.opt_p_w)}
        self.opt_p_returns = self._portfolio_returns(self.opt_p_w)
        self.opt_p_stdev = self._portfolio_stdev(self.opt_p_w)
        self.opt_p_sharpe = -1 * self._sharpe_ratio(self.opt_p_w)
        # Create dataframe containing weights, portfolio return,
        # portfolio standard deviation and sharpe ratio of the
        # optimal portfolio.
        self.opt_p_results = pd.DataFrame(
            {
                **w_dict,
                "mu_p": self.opt_p_returns,
                "sigma_p": self.opt_p_stdev,
                "sharpe": self.opt_p_sharpe,
            },
            index=[0],
        )

        return self.opt_p_results
