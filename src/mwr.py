import numpy as np
import pandas as pd
from scipy import optimize


def money_weighted_return_annualized(nav, aum, flows=None):
    """
    Calculate annualized Money-weighted return (MWR).

    Parameters
    ----------
    nav : pd.Series
        Index: pd.Timestamp
        Values: float
            Net Assets Value (NAV) of a fund/etf/portfolio.

    aum : pd.Series
        Index: pd.Timestamp
        Values: float
            Assets Under Management (AUM) of a fund/etf/portfolio.

    flows : pd.Series
        Index: pd.Timestamp
        Values: float
            Inflow(+)/Outflow(-) amounts made. If this param is not passed
            they would be estimated using the AUM and NAV returns.

    Returns
    -------
    float
    """
    if flows is None:
        flows, _ = _estimate_flows(nav=nav, aum=aum)

    cashflows = flows.mul(-1)
    cashflows[0] = -aum[0]
    cashflows[-1] += aum[-1]

    mwr = _xirr(cashflows)

    return mwr


def theoretical_mwr_annualized_with_inverted_flows(nav, aum, max_flows_weight=0.3):
    """
    Calculate theoretical annualized Money-weighted return, if the opposite
    flows were made (convert inflows to outflows, and outflows to inflows).

    Parameters
    ----------
    nav : pd.Series
        Index: pd.Timestamp
        Values: float
            Net Assets Value (NAV) of a fund/etf/portfolio.

    aum : pd.Series
        Index: pd.Timestamp
        Values: float
            Assets Under Management (AUM) of a fund/etf/portfolio.

    max_flows_weight : float
        Maximum flows weight.
        As original inflows can represent a weight greater than 100%, it is
        needed to clip them when converting to outflows (outflows > 100% do
        not make sense).
        This parameter also prevents from biasing the results due to some
        anomalous big flows.

    Returns
    -------
    float
    """
    max_flows_weight = abs(max_flows_weight)
    nav_returns = nav.pct_change().add(1.0)

    # Clip min/max flows weights
    flows_amount, flows_weight = _estimate_flows(nav=nav, aum=aum)

    flows_weight = flows_weight.clip(-max_flows_weight, max_flows_weight)

    # First calculate the theoretical AUM if the opposite flows were made
    # Opposite flows are equal in weight, not in amount
    # (for avoiding negative AUMs)
    aum_inv = aum.copy()
    flows_amount_inv = flows_amount.copy()

    for i in range(1, aum_inv.shape[0]):
        flows_amount_inv[i] = flows_weight[i] * aum_inv[i - 1] * -1
        aum_inv[i] = aum_inv[i - 1] * nav_returns[i] + flows_amount_inv[i]

    # Then calculate the MWR with this new theoretical AUM
    mwr_inv = money_weighted_return_annualized(nav=nav, aum=aum_inv, flows=flows_amount_inv)

    return mwr_inv


def theoretical_mwr_annualized(nav, aum, max_flows_weight=0.3):
    """
    Calculate theoretical annualized Money-weighted return, clipping the
    flows weight for avoiding biasing the results with anomalous big flows.

    Parameters
    ----------
    nav : pd.Series
        Index: pd.Timestamp
        Values: float
            Net Assets Value (NAV) of a fund/etf/portfolio.

    aum : pd.Series
        Index: pd.Timestamp
        Values: float
            Assets Under Management (AUM) of a fund/etf/portfolio.

    max_flows_weight : float
        Maximum flows weight.

    Returns
    -------
    float
    """
    max_flows_weight = abs(max_flows_weight)
    nav_returns = nav.pct_change().add(1.0)

    # Clip min/max flows weights
    flows_amount, flows_weight = _estimate_flows(nav=nav, aum=aum)

    flows_weight = flows_weight.clip(-max_flows_weight, max_flows_weight)

    # First calculate the theoretical AUM with the clipped flows
    aum = aum.copy()

    for i in range(1, aum.shape[0]):
        flows_amount[i] = +flows_weight[i] * aum[i - 1]
        aum[i] = aum[i - 1] * nav_returns[i] + flows_amount[i]

    # Then calculate the MWR with this new theoretical AUM
    mwr = money_weighted_return_annualized(nav=nav, aum=aum, flows=flows_amount)

    return mwr


def _estimate_flows(nav, aum):
    """
    Estimate flows: (+) inflows, (-) outflows.

    Parameters
    ----------
    nav : pd.Series
        Index: pd.Timestamp
        Values: float
            Net Assets Value (NAV) of a fund/etf/portfolio.

    aum : pd.Series
        Index: pd.Timestamp
        Values: float
            Assets Under Management (AUM) of a fund/etf/portfolio.

    Returns
    -------
    pd.Series
        Index: pd.Timestamp
        Values: float
            Estimated flows amounts.

    pd.Series
        Index: pd.Timestamp
        Values: float
            Estimated flows weights.
    """
    returns_nav = nav.pct_change()
    returns_aum = aum.pct_change()

    flows_weight = returns_aum - returns_nav
    flows_amount = flows_weight * aum.shift(1)

    return flows_amount, flows_weight


def _xirr(cashflows):
    """
    Calculate the Irregular Rate of Return (XIRR).

    Parameters
    ----------
    cashflows : pd.Series
        Index: pd.Timestamp
        Values: float
            Cashflows with sign.

    Returns
    -------
    float
    """
    # Clean values
    cashflows = cashflows[cashflows != 0]

    # Check for sign change
    if cashflows.min() * cashflows.max() >= 0:
        return np.nan

    # Set index to time delta in years
    cashflows.index = (cashflows.index - cashflows.index.min()).days / 365.0

    try:
        result = optimize.newton(lambda r: (cashflows / ((1 + r) ** cashflows.index)).sum(), x0=0, rtol=1e-4)
    except (RuntimeError, OverflowError):
        result = optimize.brentq(lambda r: (cashflows / ((1 + r) ** cashflows.index)).sum(), a=-0.999999999999999, b=100, maxiter=10 ** 4)

    if not isinstance(result, complex):
        return result
    else:
        return np.nan
