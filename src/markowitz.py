import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Markowitz:
    def __init__(self, df_returns):
        self.df_returns = df_returns
        self.n_stocks = self.df_returns.shape[1]
        self.mu_p = self.df_returns.mean() * 252
        self.cov = self.df_returns.cov() * 252

    def portfolio_returns(self, w):
        return w @ self.mu_p

    def portfolio_stdev(self, w):
        return np.sqrt(w @ self.cov @ w)

    def sharpe_ratio(self, w):
        return -1 * self.portfolio_returns(w) / self.portfolio_stdev(w)

    def simulate_portfolios(self, n_sims=1000):
        p_w = []
        p_returns = []
        p_stdev = []
        p_sharpe = []
        for n in range(n_sims):
            w = np.random.dirichlet([1] * self.n_stocks)
            p_w.append(w)
            p_returns.append(self.portfolio_returns(w))
            p_stdev.append(self.portfolio_stdev(w))
            p_sharpe.append(self.sharpe_ratio(w))

        w_df = pd.DataFrame(p_w, columns=[f"w{i}" for i in range(self.n_stocks)])
        self.df_sim = pd.DataFrame(
            {"mu_p": p_returns, "sigma_p": p_stdev, "sharpe": p_sharpe},
        )
        self.df_sim = pd.concat([w_df, self.df_sim], axis=1)
        self.df_sim["sharpe"] = -1 * self.df_sim["sharpe"]
        return self.df_sim

    def optimal_portfolio(self, obj="min-var"):
        # Initialise weights from Dirichlet(1,1,1,1) distribution.
        w0 = np.random.dirichlet([1] * self.n_stocks)
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = ((0, 1),) * self.n_stocks

        if obj == "min-var":
            obj_fun = self.portfolio_stdev
        elif obj == "max-sharpe":
            obj_fun = self.sharpe_ratio

        result = minimize(
            obj_fun, w0, method="SLSQP", constraints=constraints, bounds=bounds
        )

        self.opt_p_w = result.x
        w_dict = {f"w{i}": j for i, j in enumerate(self.opt_p_w)}
        self.opt_p_returns = self.portfolio_returns(self.opt_p_w)
        self.opt_p_stdev = self.portfolio_stdev(self.opt_p_w)
        self.opt_p_sharpe = -1 * self.sharpe_ratio(self.opt_p_w)
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