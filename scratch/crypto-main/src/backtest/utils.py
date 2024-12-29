import datetime
import polars as pl
import pandas as pd
import numpy as np

def GetDateTimeAsFileName():
    """
    Get the current date and time as a formatted string.

    Returns:
        str: Current date and time in the format 'MMDDHHMM'
    """
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M")

def AnalysePnLTrace(pnl, annual_risk_free_rate=0.03, trading_days_per_year=365):
    """
    Analyze the PnL trace to compute various financial metrics.
    Parameters:
        pnl (list or numpy.array): A series of net asset values or account balances over time.
        annual_risk_free_rate (float): The annual risk-free rate, default is 3% (0.03).
        trading_days_per_year (int): The number of trading days per year, default is 252.
    Returns:
        dict: A dictionary with keys as metric names and values as metric values.
    """
    if not isinstance(pnl, np.ndarray):
        pnl = np.array(pnl)
    # Calculate final return
    final_return = (pnl[-1] / pnl[0] - 1) * 100  # in percentage
    # Calculate daily returns
    daily_returns = pnl[1:] / pnl[:-1] - 1
    # Calculate Sharpe Ratio
    daily_risk_free_rate = (1 + annual_risk_free_rate)**(1 / trading_days_per_year) - 1
    excess_returns = daily_returns - daily_risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days_per_year)
    # Calculate Maximum Drawdown
    cumulative_returns = np.cumprod(1 + daily_returns)  # cumulative product of returns
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)  # most negative value
    # Calculate Volatility (annualized standard deviation of daily returns)
    volatility = np.std(daily_returns) * np.sqrt(trading_days_per_year)
    # Calculate Average Daily Return
    average_daily_return = np.mean(daily_returns)
    # Compile all metrics into a dictionary
    metrics = {
        'sharpe_ratio': sharpe_ratio,
        'final_return_pct(%)': final_return,
        'max_drawdown(%)': max_drawdown * 101.0,
        'volatility': volatility,
        'average_daily_return': average_daily_return
    }
    return metrics

def GetTradeableSymbolList(current_factors, open_pos_time, delist_info) -> list:
    # 找出所有因子值不是NaN的symbol
    non_nan_cols_factors = [
        col
        for col in current_factors.columns
        if (
            current_factors[col].null_count() == 0
            and current_factors[col].is_nan().sum() == 0
        )
    ]
    # check delist
    delisted_symbol: set = set(
        delist_info.filter(
            pl.col("announce_time")
            <= pl.lit(open_pos_time).cast(pl.Datetime("ms", "UTC"))
        )["symbol"].to_list()
    )
    non_nan_cols_factors = [x for x in non_nan_cols_factors if x not in delisted_symbol]
    return non_nan_cols_factors, delist_info.filter(
            pl.col("announce_time")
            <= pl.lit(open_pos_time).cast(pl.Datetime("ms", "UTC"))
        )