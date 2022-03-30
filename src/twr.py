import pandas as pd


def time_weighted_return_annualized(nav):
    """
    Calculate annualized Time-weighted return (TWR).

    Parameters
    ----------
    nav : pd.Series
        Index: pd.Timestamp
        Values: float
            Net Assets Value (NAV) of a fund/etf/portfolio.

    Returns
    -------
    float
    """
    days = (nav.index[-1] - nav.index[0]).days
    annual_factor = 365.0 / days

    twr = ((nav[-1] / nav[0]) ** annual_factor) - 1

    return twr
