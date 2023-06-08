import requests
import config
import pandas as pd
import numpy as np
import logging
import datetime

# Set logger
logging.basicConfig(level=logging.INFO)

class MarketData:
    """
    A class to retrieve EOD stock market data and yield their daily log-returns.

    Attributes
    ----------
    symbols : str
        Stock ticker symbols e.g. "AAPL,NVDA"
    date_from : str
        From date in yyyy-mm-dd format.
    date_to : str
        To date in yyyy-mm-dd format.

    Methods
    -------
    raw_data()
        Extracts raw data in descending order for all specified stocks.
    """

    def __init__(self, symbols, date_from, date_to):
        self.symbols = symbols
        self.date_from = date_from
        self.date_to = date_to

    def raw_data(self):
        """
        Retrieves EOD raw data for user-specified stocks within a time-window.

        Parameters
        ----------
            None

        Returns
        -------
        raw_df : pandas.DataFrame
            Dataframe including EOD information about the selected stocks:
                - open
                - high
                - low
                - close
                - volume
                - adj_high
                - adj_low
                - adj_close
                - adj_open
                - adj_volume
                - split_factor
                - dividend
                - symbol
                - exchange
                - date
        """
        base_url = "http://api.marketstack.com/v1/eod"
        params = {
            "access_key": config.API_KEY,
            "symbols": self.symbols,
            "sort": "DESC",
            "date_from": self.date_from,
            "date_to": self.date_to,
            "limit": 1000
        }
        logging.info(f"Retrieving data from {self.date_from} to {self.date_to} for {self.symbols} stocks.")
        api_result = requests.get(base_url, params)
        api_response = api_result.json()
        self.raw_df = pd.DataFrame.from_dict(api_response["data"])
        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"]).dt.strftime("%Y-%m-%d")
        logging.info("Data retrieved.")
        return self.raw_df

    def transform_data(self):
        """
        Calculates daily log-returns for stocks.

        Parameters
        ----------
        None

        Returns
        -------
        df_trans : pd.DataFrame
            Transformed dataframe containing the daily log-returns for the stocks.
        """
        self.df_trans = pd.pivot_table(
            self.raw_df, values="adj_close", index="date", columns="symbol"
        )
        logging.info("Transforming data.")
        self.df_trans = np.log(self.df_trans / self.df_trans.shift(1))
        self.df_trans.dropna(inplace=True)
        logging.info("Data transformed.")
        return self.df_trans
    
if __name__ == "__main__":
    # Define tickers for Nvidia, Johnson&Johnson, Procter&Gamble, and JPMorgan.
    symbols = "NVDA,AAPL,TSLA,MSFT"
    # Capture one-year worth of data.
    date_from = "2022-06-02" 
    date_to = "2023-06-02"
    # Instantiate object from MarketData class.
    md = MarketData(symbols, date_from, date_to)
    # Retrieve raw data.
    df_raw = md.raw_data()
    # Retrieve transformed log-return data.
    df_trans = md.transform_data()
    # Save both datasets as .csv's for later use.
    df_raw.to_csv("./stocks_raw.csv", index=False)
    df_trans.to_csv("./stocks_returns.csv")