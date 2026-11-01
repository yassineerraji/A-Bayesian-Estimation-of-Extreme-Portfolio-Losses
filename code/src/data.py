"""Load the processed daily log-returns."""

import pandas as pd


def load_returns(cfg):
    """Return the daily log-returns (decimals) as a pandas Series indexed by date."""
    return pd.read_csv(cfg["paths"]["returns"], index_col="date", parse_dates=True)["r"]
