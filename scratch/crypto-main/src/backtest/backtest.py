import polars as pl
import numpy as np
import pandas as pd
import warnings
import logging
import os
from typing import List, Optional
from .get_single_backtest_pnl import GetSinglePnL
from .get_rolling_backtest_pnl import GetRollingPnL
from .config import BacktestConfig
from .pnl_entry import PnLEntry
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# Suppress warnings
warnings.filterwarnings("ignore")
pl.Config.set_ascii_tables(True)


def print_versions():
    """Print versions of key dependencies."""
    print(f"Polars version: {pl.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")


def setup_logger(config: BacktestConfig, log_level: int = logging.DEBUG) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger()
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Configure new handler
    logging.basicConfig(
        filename=config.log_path,
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    return logger


def configure_polars(config: BacktestConfig):
    """Configure Polars display settings."""
    pl.Config.set_tbl_rows(config.show_row_num)



def Backtest(config: BacktestConfig, history_data: pl.DataFrame, if_rolling: bool, log_level: int = logging.DEBUG) -> None:
    # print_versions()

    # Setup logging
    logger = setup_logger(config, log_level)
    print(f"Run Backtest. Logging configured. Log file: {config.log_path}")

    # Configure Polars, make it prints more rows
    configure_polars(config)

    if if_rolling:
        pnl_records, analysis, sub_pnl_traces_list = GetRollingPnL(
                    logger=logger,
                    result_hour=history_data,
                    FACTOR_NAME=config.trade_factor_name,
                    config=config
                )
        return pnl_records, analysis, sub_pnl_traces_list

    else:
        pnl, metrics, _ = GetSinglePnL(
            logger=logger,
            result_hour=history_data,
            FACTOR_NAME=config.trade_factor_name,
            config = config,
        )
        
        return pnl, metrics, None

def plot_simple_pnl(pnl_entries: List[PnLEntry], config: BacktestConfig, sub_pnl_traces: Optional[List[np.ndarray]] = None, metrics: Optional[dict] = None):
    """
    Plot PnL traces with main PnL, optional sub-traces, and metrics information.
    
    Args:
        pnl_entries: List of PnLEntry for the main PnL trace
        config: Backtest configuration
        sub_pnl_traces: Optional list of numpy arrays containing sub-traces
        metrics: Optional dictionary containing performance metrics
    """
    # Create figure with extra height for metrics
    plt.figure(figsize=(12, 9))
    
    # Create main plot subplot
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    
    # Extract timestamps and values for main PnL
    timestamps = [entry.timestamp for entry in pnl_entries]
    values = [entry.value for entry in pnl_entries]
    
    # Plot main PnL trace
    ax.plot(timestamps, values, 'b-', linewidth=2.5, label='Overall PnL')
    
    # Plot sub-traces if provided
    if sub_pnl_traces is not None:
        # Generate distinct colors for sub-traces
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sub_pnl_traces)))
        
        for i, sub_trace in enumerate(sub_pnl_traces):
            # Ensure sub_trace length matches timestamps
            if len(sub_trace) <= len(timestamps):
                ax.plot(timestamps[:len(sub_trace)], sub_trace, 
                        color=colors[i], linewidth=1, alpha=0.5,
                        linestyle='--', label=f'Sub-trace {i+1}')
            else:
                print(f"Warning: Sub-trace {i+1} length mismatch")
    
    # Enhance plot formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    
    plt.title(f'Backtest PnL: {os.path.basename(config.input_df_path)}\n', pad=20)
    ax.set_ylabel('PnL Value', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add metrics information if provided
    if metrics:
        # Create a text box for metrics
        metrics_text = "Overall Performance Metrics:\n\n"
        # Format each metric with proper alignment
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metrics_text += f"{key:<20}: {value:,.4f}\n"
            else:
                metrics_text += f"{key:<20}: {value}\n"
        
        # Add text box in the bottom portion of the figure
        plt.figtext(0.1, 0, metrics_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    family='monospace')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    
    # Save plot with increased bottom margin for metrics
    plt.savefig(config.plot_path, bbox_inches='tight', dpi=300, 
                pad_inches=0.5)  # Increased padding for metrics
    plt.close()

def main_backtest(config: BacktestConfig, if_rolling: bool = True):
    config.validate()
    print(f"main_backtest ==== read result hour path: {config.input_df_path}")
    
    input_df = pl.read_parquet(config.input_df_path)

    # convert from timestamp to datetime format
    input_df = input_df.with_columns(
        pl.from_epoch(pl.col("open_time"), time_unit="ms")
        .cast(pl.Datetime("ms"))
        .alias("open_time"),
        pl.from_epoch(pl.col("close_time"), time_unit="ms")
        .cast(pl.Datetime("ms"))
        .alias("close_time"),
    )

    # only use the data after start_date
    if config.start_date is not None:
        input_df = input_df.filter(pl.col("open_time") >= config.start_date)
    input_df = input_df.sort("open_time")

    compound_column_name = "linear_compound_factor"
    FACTOR_NAME = compound_column_name + f"_{config.update_position_time}day"

    if FACTOR_NAME not in input_df.columns:
        print(f"Factor {FACTOR_NAME} not found in data")
        return

    cur_pnl, cur_metric, sub_pnl_traces_list = Backtest(
        config=config,
        history_data=input_df.filter(pl.col(FACTOR_NAME).is_not_null()),
        if_rolling = if_rolling,
    )

    plot_simple_pnl(cur_pnl, config, sub_pnl_traces_list, metrics=cur_metric)
    print(f'Metrics: {cur_metric}')

def show_plot(config: BacktestConfig):
    """Display the saved plot in a notebook cell"""
    from IPython.display import Image
    import os
    import time
    
    # Wait a brief moment for file system sync
    time.sleep(0.5)
    
    if not os.path.exists(config.plot_path):
        print(f"Plot not found at: {config.plot_path}")
        return
        
    try:
        return Image(filename=config.plot_path, width=1000)
    except Exception as e:
        print(f"Error displaying plot: {e}")