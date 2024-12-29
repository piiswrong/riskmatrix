from typing import List, Tuple
import polars as pl
import numpy as np
import logging
from .get_single_backtest_pnl import GetSinglePnL
from .pnl_entry import PnLEntry
from .config import BacktestConfig
from .utils import AnalysePnLTrace
from typing import Optional

def slice_data_for_rolling(
    result_hour: pl.DataFrame,
    unique_time: pl.DataFrame,
    start_index: int
) -> pl.DataFrame:
    """
    Slice data for rolling calculation.
    """
    cur_unique_time = unique_time.slice(start_index)
    return result_hour.filter(pl.col("open_time").is_in(cur_unique_time))

def calculate_single_rolling_pnl(
    logger: logging.Logger,
    result_hour: pl.DataFrame,
    config: BacktestConfig,
    start_index: int,
    pnl_length: int,
) -> List[float]:
    """
    Calculate PnL for a single rolling period.
    """
    cur_result_hour = slice_data_for_rolling(result_hour, 
                                           result_hour.select(pl.col("open_time").unique()).sort("open_time"),
                                           start_index)
    
    pnl_records, _, _ = GetSinglePnL(
        logger=logger,
        result_hour=cur_result_hour,
        FACTOR_NAME=config.trade_factor_name,
        config=config
    )
    
    # Convert PnL entries to values and pad with initial cash
    pnl_values = [entry.value for entry in pnl_records]
    padded_pnl = np.insert(pnl_values, 0, [config.start_cash] * start_index)
    
    if len(padded_pnl) != pnl_length:
        raise ValueError(f"PnL length mismatch: expected {pnl_length}, got {len(padded_pnl)}")
    
    return padded_pnl

def GetRollingPnL(
    logger: logging.Logger,
    result_hour: pl.DataFrame,
    FACTOR_NAME: str,
    config: BacktestConfig,
    delist_info: Optional[pl.DataFrame] = None
) -> Tuple[List[PnLEntry], str, str]:
    """
    Calculate rolling PnL across multiple periods.
    
    Args:
        logger: Logger instance
        result_hour: Trading data
        FACTOR_NAME: Name of the trading factor
        config: Backtest configuration
        delist_info: Optional delisting information
    
    Returns:
        Tuple[List[PnLEntry], str, str]: PnL entries, analysis metrics, and statistics
    """
    # Get unique timestamps
    unique_time = result_hour.select(pl.col("open_time").unique()).sort("open_time")
    update_position_time = config.update_position_time
    
    # Initialize arrays for calculation
    pnl_length = unique_time.shape[0] - 1  # last day doesn't have PnL
    sum_pnl: np.ndarray = np.zeros(pnl_length)
    sub_pnl_trace_list: List[np.ndarray] = []
    
    # Calculate PnL for each rolling period
    for i in range(update_position_time):
        try:
            cur_pnl = calculate_single_rolling_pnl(
                logger,
                result_hour,
                config,
                i,
                pnl_length
            )
            
            sum_pnl += cur_pnl
            sub_pnl_trace_list.append(cur_pnl)
            logger.info(f'Rolling {i}: PnL length={len(cur_pnl)}, Analysis={AnalysePnLTrace(cur_pnl)}')
            print (f' ==== Rolling {i}: Analysis={AnalysePnLTrace(cur_pnl)}')
            
        except Exception as e:
            logger.error(f"Error in rolling period {i}: {str(e)}")
            continue
    
    # Calculate average PnL
    avg_pnl = sum_pnl / update_position_time
    
    # Convert average PnL to PnLEntry list
    pnl_records: List[PnLEntry] = []
    timestamps = unique_time["open_time"].to_list()
    for i, value in enumerate(avg_pnl):
        if i < len(timestamps) - 1:  # Ensure we don't exceed timestamp bounds
            pnl_records.append(PnLEntry(timestamps[i], float(value)))
    
    # Generate analysis
    analysis = AnalysePnLTrace([entry.value for entry in pnl_records])
    
    return pnl_records, analysis, sub_pnl_trace_list
