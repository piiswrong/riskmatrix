import polars as pl
import os
import numpy as np
from math import fabs
from typing import Dict, Optional, Tuple
from .utils import AnalysePnLTrace, GetTradeableSymbolList
from .pnl_entry import PnLEntry
from .config import BacktestConfig

# print trade details
def GetTradeDetailGivenTimeNSymbol(
    cur_close_time, cur_open_pos_time, cur_trade_list, future_n_day_open_ret, all_open_price_when_open_pos, current_factors
) -> pl.DataFrame:
    if len(cur_trade_list) == 0:
        return pl.DataFrame(
            {
                "symbol": [],
                "factor": [],
                "open_price": [],
                "open_price_volatility": [],
                "future_ret": [],
            }
        )

    cur_future_ret = future_n_day_open_ret.filter(
        pl.col("open_time") == cur_open_pos_time
    ).select(pl.col(cur_trade_list))

    cur_trade_open_price = all_open_price_when_open_pos.select(
        pl.col(cur_trade_list)
    )

    cur_trade_factor_val = current_factors.select(
        pl.col(cur_trade_list)
    )

    assert cur_future_ret.shape == cur_trade_open_price.shape
    log_df = pl.DataFrame(
        {
            "symbol": cur_trade_list,
            "factor": cur_trade_factor_val.to_numpy()[0],
            "open_price": cur_trade_open_price.to_numpy()[0],
            "future_ret": cur_future_ret.to_numpy()[0],
        }
    )

    log_df = log_df.sort("factor")
    log_df = log_df.fill_nan(None)
    log_df = log_df.select(
        pl.col(
            [
                "symbol",
                "factor",
                "open_price",
                "future_ret",
            ]
        )
    )
    log_df = log_df.sort("future_ret")
    return log_df


def prepare_data(result_hour: pl.DataFrame, FACTOR_NAME: str, update_position_time: int) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Prepare and transform input data for backtesting."""
    FUT_N_DAY_RET_COL_NAME = f"open_price_fut_{update_position_time}day_ret"
    
    # Get future returns
    future_n_day_open_ret = (
        result_hour[["open_time", "symbol", FUT_N_DAY_RET_COL_NAME]]
        .pivot(index="open_time", columns="symbol", values=FUT_N_DAY_RET_COL_NAME)
        .sort("open_time")
    )

    # Initialize long value scale
    result_hour = result_hour.with_columns(
        pl.lit(0).alias(f"long_value_scale_{update_position_time}day")
    )
    long_scale = (
        result_hour[["close_time", "symbol", f"long_value_scale_{update_position_time}day"]]
        .pivot(index="close_time", columns="symbol", values=f"long_value_scale_{update_position_time}day")
        .sort("close_time")
    )

    # Get factors
    # 因为此时的因子值是在close_time的时候计算的，所以要用close_time
    factors = (
        result_hour[["close_time", "symbol", FACTOR_NAME]]
        .pivot(index="close_time", columns="symbol", values=FACTOR_NAME)
        .sort("close_time")
    )

    # Get open prices
    # 每根k线的开盘价
    open_df = (
        result_hour[["open_time", "symbol", "open"]]
        .pivot(index="open_time", columns="symbol", values="open")
        .sort("open_time")
    )

    # Get volume data
    bar_close_vol = (
        result_hour[["close_time", "symbol", "volume"]]
        .pivot(index="close_time", columns="symbol", values="volume")
        .sort("close_time")
    )

    return future_n_day_open_ret, long_scale, factors, open_df, bar_close_vol

def handle_early_cut_loss(
    cur_position: pl.DataFrame,
    cur_pos_open_price: pl.DataFrame,
    all_open_price_when_open_pos: pl.DataFrame,
    cur_hold_long_symbol_list: list,
    cur_hold_short_symbol_list: list,
    cash: float,
    cut_loss_pct_threshold: float,
    commission: float,
    eps: float,
    logger
) -> Tuple[pl.DataFrame, float]:
    """Handle early cut loss for positions."""
    each_symbol_change_pct = (all_open_price_when_open_pos / cur_pos_open_price - 1) * 100.0
    
    for symbol in cur_hold_long_symbol_list + cur_hold_short_symbol_list:
        this_symbol_change_pct = each_symbol_change_pct[symbol].to_numpy()[0]
        this_symbol_hold_amount = cur_position[symbol].to_numpy()[0]

        if ((this_symbol_hold_amount > eps and this_symbol_change_pct < -cut_loss_pct_threshold) or
            (this_symbol_hold_amount < -eps and this_symbol_change_pct > cut_loss_pct_threshold)):
            
            this_symbol_close_pos_price = all_open_price_when_open_pos[symbol].to_numpy()[0]
            cash += this_symbol_close_pos_price * this_symbol_hold_amount
            
            this_symbol_trading_value = fabs(this_symbol_hold_amount) * this_symbol_close_pos_price
            cash -= this_symbol_trading_value * commission
            
            cur_position = cur_position.with_columns(pl.lit(0).alias(symbol))
            
            logger.debug(f"Cut loss for {symbol}: change_pct={this_symbol_change_pct}")
    
    return cur_position, cash

def handle_delisted_symbols(
    cur_position: pl.DataFrame,
    delist_info: pl.DataFrame,
    cur_close_time: np.datetime64,
    bar_open_time_to_close_pos: np.datetime64,
    all_open_price_when_open_pos: pl.DataFrame,
    cash: float,
    logger
) -> Tuple[pl.DataFrame, float]:
    """Handle delisted symbols in the portfolio."""
    delisted_symbol: list = delist_info.filter(
        (pl.col("announce_time") >= pl.lit(cur_close_time).cast(pl.Datetime("ms", "UTC"))) & 
        (pl.col("end_trade_time") <= pl.lit(bar_open_time_to_close_pos).cast(pl.Datetime("ms", "UTC")))
    )["symbol"].to_list()

    for each_delist_symbol in delisted_symbol:
        if each_delist_symbol in cur_position.columns and cur_position[each_delist_symbol].to_numpy()[0] != 0:
            current_position = float(cur_position[each_delist_symbol].to_numpy()[0])
            cur_position = cur_position.with_columns(pl.lit(0.0).alias(each_delist_symbol))
            next_bar_open_price = all_open_price_when_open_pos[each_delist_symbol].to_numpy()[0]
            cash += next_bar_open_price * current_position
            
            logger.warning(f"Closed delisted position: {each_delist_symbol} at {cur_close_time}")
    
    return cur_position, cash

def TradeableSymbolFilter(
    tradeable_symbol_list: list,
    cur_bar_close_vol: pl.DataFrame,
    vol_filter_ratio: float,
    cur_close_time: np.datetime64,
    logger
) -> list:
    """
    Filter tradeable symbols based on volume threshold.
    
    Args:
        tradeable_symbol_list (list): List of initially tradeable symbols
        cur_bar_close_vol (pl.DataFrame): Volume data for the current bar
        vol_filter_ratio (float): Percentile threshold for volume filtering
        cur_close_time (np.datetime64): Current close time for logging
        logger: Logger instance for debug information
    
    Returns:
        list: Filtered list of tradeable symbols that meet volume threshold
    """
    # Collect volumes for all tradeable symbols
    vol_for_all_symbols = []
    for symbol in tradeable_symbol_list:
        vol_for_all_symbols.append(cur_bar_close_vol[symbol].to_list()[0])
    
    # Calculate volume threshold
    vol_threshold = np.percentile(vol_for_all_symbols, vol_filter_ratio)
    logger.debug(f'close time {cur_close_time} == vol_threshold: {vol_threshold}')
    
    # Filter symbols based on volume threshold
    tradeable_symbol_list_with_large_vol = []
    for symbol in tradeable_symbol_list:
        assert len(cur_bar_close_vol[symbol].to_list()) > 0, f'{cur_bar_close_vol[symbol].to_list()} == {symbol}'
        if cur_bar_close_vol[symbol].to_list()[0] > vol_threshold:
            tradeable_symbol_list_with_large_vol.append(symbol)
            
    logger.debug(f'{cur_close_time} == all trade symbol {len(tradeable_symbol_list)} == remain {len(tradeable_symbol_list_with_large_vol)} after VOL filter')
    
    return tradeable_symbol_list_with_large_vol

def sum_row(df: pl.DataFrame) -> float:
    return df.select(
        pl.sum_horizontal(
            # pl.all().filter(pl.col.is_numeric())
            pl.all()
        ).fill_null(0)
    ).item()


def prepare_factor_data(current_factors: pl.DataFrame, tradeable_symbol_list: list) -> pl.DataFrame:
    """
    准备因子数据，将可交易股票的因子值提取出来并进行验证
    
    Args:
        current_factors: 当前时间点的因子数据
        tradeable_symbol_list: 可交易的股票列表
    
    Returns:
        pl.DataFrame: 处理后的因子数据框
    """
    long_factors = current_factors.select(tradeable_symbol_list).melt(
        id_vars=[],
        value_vars=tradeable_symbol_list,
        variable_name="symbol",
        value_name="factor_value",
    )

    # 验证因子值的有效性
    assert long_factors["factor_value"].is_null().sum() == 0, "factor_value column contains null values"
    assert long_factors["factor_value"].is_nan().sum() == 0, "factor_value column contains NaN values"
    
    return long_factors.sort("factor_value")

def create_rank_labels(sorted_long_factors: pl.DataFrame, trade_with_rank: int, group_size: int) -> pl.DataFrame:
    """
    根据交易策略创建排名或分组标签
    
    Args:
        sorted_long_factors: 已排序的因子数据
        trade_with_rank: 是否使用排名交易
        group_size: 分组大小
    
    Returns:
        pl.DataFrame: 添加了排名或分组标签的数据框
    """
    if trade_with_rank != 0:
        # 创建正向排名
        rank_labels = (pl.arange(0, sorted_long_factors.height)).cast(pl.UInt32)
        sorted_long_factors = sorted_long_factors.with_columns(rank_labels.alias("rank"))
        
        # 创建反向排名
        max_rank = sorted_long_factors.height - 1
        reverse_rank_labels = (max_rank - rank_labels).cast(pl.UInt32)
        sorted_long_factors = sorted_long_factors.with_columns(reverse_rank_labels.alias("reverse_rank"))
    else:
        # 创建分组标签
        group_labels = (pl.arange(0, sorted_long_factors.height) / group_size).cast(pl.UInt32)
        sorted_long_factors = sorted_long_factors.with_columns(group_labels.alias("group"))
    
    return sorted_long_factors

def select_trading_stocks(
    sorted_long_factors: pl.DataFrame,
    trade_with_rank: int,
    cur_trade_num: int,
    long_trade_rank_ratio: float,
    short_trade_rank_ratio: float,
    group_num: int,
    long_factor_combination_list: list
) -> Tuple[list, list]:
    """
    根据排名或分组选择交易股票
    
    Args:
        sorted_long_factors: 带有排名或分组标签的因子数据
        trade_with_rank: 是否使用排名交易
        cur_trade_num: 当前交易数量
        long_trade_rank_ratio: 多头交易比例
        short_trade_rank_ratio: 空头交易比例
        group_num: 分组数量
        long_factor_combination_list: 多头组合列表
    
    Returns:
        Tuple[list, list]: 多头和空头股票列表
    """
    if trade_with_rank != 0:
        assert (long_trade_rank_ratio + short_trade_rank_ratio) * cur_trade_num <= sorted_long_factors.height, \
            "total of long & short should not exceed total symbol number"
        
        cur_rank_column = "rank" if trade_with_rank > 0 else "reverse_rank"
        long_stock_set = sorted_long_factors.filter(
            pl.col(cur_rank_column) < cur_trade_num * long_trade_rank_ratio
        )
        short_stock_set = sorted_long_factors.filter(
            pl.col(cur_rank_column) > sorted_long_factors.height - 1 - cur_trade_num * short_trade_rank_ratio
        )
    else:
        short_combination_list = [group_num - 1 - x for x in long_factor_combination_list]
        long_stock_set = sorted_long_factors.filter(pl.col("group").is_in(long_factor_combination_list))
        short_stock_set = sorted_long_factors.filter(pl.col("group").is_in(short_combination_list))
    
    long_symbol_list = long_stock_set.select("symbol").unique().to_pandas().squeeze().to_list()
    short_symbol_list = short_stock_set.select("symbol").unique().to_pandas().squeeze().to_list()
    
    return long_symbol_list, short_symbol_list

def calculate_position_ratios(latest_pnl: float, overall_leverage: float, long_symbol_list: list, 
                            short_symbol_list: list, cur_close_time: np.datetime64, logger) -> Tuple[float, float, Dict, Dict]:
    """
    计算多空仓位的总价值和每个股票的仓位比例
    
    Args:
        latest_pnl: 最新的PnL
        overall_leverage: 总杠杆
        long_symbol_list: 多头股票列表
        short_symbol_list: 空头股票列表
        cur_close_time: 当前收盘时间
        logger: 日志记录器
    
    Returns:
        Tuple[float, float, Dict, Dict]: 多头总价值，空头总价值，多头仓位比例字典，空头仓位比例字典
    """
    each_side_symbol_total_val = latest_pnl * overall_leverage
    # each_side_symbol_total_val = 100000.0  # 单边的总价值
    logger.info(f"each_side_symbol_total_val: {each_side_symbol_total_val} ====  last pnl: {latest_pnl}")
    
    # 多空的总仓位比例，二者之和为2
    long_value_ratio = 1.0  # 目前固定为多空都是1
    short_value_ratio = 1.0
    assert long_value_ratio + short_value_ratio == 2.0, "sum ratio should be 2.0"
    
    logger.info(f"cur close time: {cur_close_time} == long_value_ratio: {long_value_ratio} == short_value_ratio: {short_value_ratio}")
    
    total_long_pos_value = each_side_symbol_total_val * long_value_ratio
    total_short_pos_value = -each_side_symbol_total_val * short_value_ratio
    
    # 初始化仓位比例字典
    long_symbol_pos_ratio_dict = {symbol: 1.0 / len(long_symbol_list) for symbol in long_symbol_list}
    short_symbol_pos_ratio_dict = {symbol: 1.0 / len(short_symbol_list) for symbol in short_symbol_list}
    
    return total_long_pos_value, total_short_pos_value, long_symbol_pos_ratio_dict, short_symbol_pos_ratio_dict

def get_closed_positions(cur_position: pl.DataFrame, next_step_position: pl.DataFrame, eps: float) -> Tuple[list, list]:
    """
    找出所有之前交易但现在关闭的仓位
    
    Args:
        cur_position: 当前仓位
        next_step_position: 下一步仓位
        eps: 误差容限
    
    Returns:
        Tuple[list, list]: 关闭的多头列表和空头列表
    """
    close_long_symbol_list = []
    close_short_symbol_list = []
    
    for cur_symbol in cur_position.columns:
        prev_step_pos_num = cur_position[cur_symbol].to_numpy()[0]
        next_step_pos_num = next_step_position[cur_symbol].to_numpy()[0]
        if prev_step_pos_num > eps and fabs(next_step_pos_num) < eps:
            close_long_symbol_list.append(cur_symbol)
        if prev_step_pos_num < -eps and fabs(next_step_pos_num) < eps:
            close_short_symbol_list.append(cur_symbol)
            
    return close_long_symbol_list, close_short_symbol_list

def adjust_position_ratios_by_factor(long_trade_detail: pl.DataFrame, short_trade_detail: pl.DataFrame, 
                                   logger) -> Tuple[Dict, Dict]:
    """
    根据因子调整仓位比例
    
    Args:
        long_trade_detail: 多头交易详情
        short_trade_detail: 空头交易详情
        logger: 日志记录器
    
    Returns:
        Tuple[Dict, Dict]: 调整后的多头和空头仓位比例字典
    """
    # 填充空值
    long_trade_detail = long_trade_detail.with_columns(pl.col("pos_factors_ratio").fill_null(0))
    short_trade_detail = short_trade_detail.with_columns(pl.col("pos_factors_ratio").fill_null(0))
    
    # 计算总和并归一化
    sum_long_pos_factor_ratio = long_trade_detail["pos_factors_ratio"].sum()
    sum_short_pos_factor_ratio = short_trade_detail["pos_factors_ratio"].sum()
    
    long_trade_detail = long_trade_detail.with_columns(pl.col("pos_factors_ratio") / pl.lit(sum_long_pos_factor_ratio))
    short_trade_detail = short_trade_detail.with_columns(pl.col("pos_factors_ratio") / pl.lit(sum_short_pos_factor_ratio))
    
    # 转换为字典
    new_long_symbol_pos_ratio_dict = {
        item["symbol"]: item["pos_factors_ratio"]
        for item in long_trade_detail.select(pl.col(["symbol", "pos_factors_ratio"])).to_dicts()
    }
    new_short_symbol_pos_ratio_dict = {
        item["symbol"]: item["pos_factors_ratio"]
        for item in short_trade_detail.select(pl.col(["symbol", "pos_factors_ratio"])).to_dicts()
    }
    
    return new_long_symbol_pos_ratio_dict, new_short_symbol_pos_ratio_dict

def validate_trade_details(long_trade_detail: pl.DataFrame, short_trade_detail: pl.DataFrame, logger):
    """
    验证交易详情数据的有效性
    
    Args:
        long_trade_detail: 多头交易详情
        short_trade_detail: 空头交易详情
        logger: 日志记录器
    """
    if long_trade_detail['future_ret'].is_null().sum() != 0:
        logger.warning("long_trade_detail['future_ret'] contains null")
    if short_trade_detail['future_ret'].is_null().sum() != 0:
        logger.warning("short_trade_detail['future_ret'] contains null")

def analyze_and_log_trading_details(
    cur_position: pl.DataFrame,
    next_step_position: pl.DataFrame,
    cur_close_time: np.datetime64,
    next_bar_open_time: np.datetime64,
    long_symbol_list: list,
    short_symbol_list: list,
    future_n_day_open_ret: pl.DataFrame,
    all_open_price_when_open_pos: pl.DataFrame,
    current_factors: pl.DataFrame,
    use_factor_pos_ratio: bool,
    long_symbol_pos_ratio_dict: Dict,
    short_symbol_pos_ratio_dict: Dict,
    eps: float,
    logger
) -> Tuple[Dict, Dict]:
    """
    分析并记录交易详情，包括未来收益和仓位调整
    
    Args:
        cur_position: 当前仓位
        next_step_position: 下一步仓位
        cur_close_time: 当前收盘时间
        next_bar_open_time: 下一个开盘时间
        long_symbol_list: 多头股票列表
        short_symbol_list: 空头股票列表
        future_n_day_open_ret: 未来收益数据
        all_open_price_when_open_pos: 开仓价格数据
        current_factors: 当前因子数据
        use_factor_pos_ratio: 是否使用因子调整仓位
        long_symbol_pos_ratio_dict: 多头仓位比例字典
        short_symbol_pos_ratio_dict: 空头仓位比例字典
        eps: 误差容限
        logger: 日志记录器
    
    Returns:
        Tuple[Dict, Dict]: 更新后的多头和空头仓位比例字典
    """
    # 找出关闭的仓位
    close_long_symbol_list, close_short_symbol_list = get_closed_positions(
        cur_position, next_step_position, eps
    )
    
    # 获取交易详情
    long_trade_detail = GetTradeDetailGivenTimeNSymbol(
        cur_close_time,
        next_bar_open_time,
        long_symbol_list,
        future_n_day_open_ret,
        all_open_price_when_open_pos,
        current_factors
    )
    short_trade_detail = GetTradeDetailGivenTimeNSymbol(
        cur_close_time,
        next_bar_open_time,
        short_symbol_list,
        future_n_day_open_ret,
        all_open_price_when_open_pos,
        current_factors
    )
    
    # 计算平均值
    long_detail_mean = long_trade_detail.select(pl.col("*").exclude("symbol").mean())
    short_detail_mean = short_trade_detail.select(pl.col("*").exclude("symbol").mean())
    
    # 如果需要，根据因子调整仓位
    if use_factor_pos_ratio:
        new_long_ratio_dict, new_short_ratio_dict = adjust_position_ratios_by_factor(
            long_trade_detail, short_trade_detail, logger
        )
        
        logger.debug(f"long pos ratio dict: {long_symbol_pos_ratio_dict} === new {new_long_ratio_dict}")
        logger.debug(f"short pos ratio dict: {short_symbol_pos_ratio_dict} ==== new {new_short_ratio_dict}")
        
        long_symbol_pos_ratio_dict = new_long_ratio_dict
        short_symbol_pos_ratio_dict = new_short_ratio_dict
    
    # 记录交易详情
    logger.debug(f"trade detail for long : {long_trade_detail} \n {long_detail_mean}")
    logger.debug(f"trade detail for short: {short_trade_detail} \n {short_detail_mean}")
    
    # 验证交易详情
    validate_trade_details(long_trade_detail, short_trade_detail, logger)
    
    return long_symbol_pos_ratio_dict, short_symbol_pos_ratio_dict

def calculate_next_step_market_value(
    all_symbols: list,
    long_symbol_list: list,
    short_symbol_list: list,
    total_long_pos_value: float,
    total_short_pos_value: float,
    long_symbol_pos_ratio_dict: Dict,
    short_symbol_pos_ratio_dict: Dict,
    logger
) -> pl.DataFrame:
    """
    计算下一步每个股票的市场价值
    """
    next_step_value = pl.DataFrame({col: [0] for col in all_symbols})
    
    # Set long positions
    for symbol in long_symbol_list:
        assert symbol in next_step_value.columns, f"Symbol {symbol} not found in columns"
        next_step_value = next_step_value.with_columns(
            pl.lit(total_long_pos_value * long_symbol_pos_ratio_dict[symbol]).alias(symbol)
        )
    
    # Set short positions
    for symbol in short_symbol_list:
        assert symbol in next_step_value.columns, f"Symbol {symbol} not found in columns"
        next_step_value = next_step_value.with_columns(
            pl.lit(total_short_pos_value * short_symbol_pos_ratio_dict[symbol]).alias(symbol)
        )
    
    logger.debug(f"next_step_value: {next_step_value}")
    return next_step_value

def calculate_position_changes(
    cur_position: pl.DataFrame,
    next_step_value: pl.DataFrame,
    all_open_price_when_open_pos: pl.DataFrame,
    commission: float,
    logger
) -> Tuple[pl.DataFrame, pl.DataFrame, float, float]:
    """
    计算仓位变动和现金变化
    """
    # 计算下一步仓位
    next_step_position = next_step_value / all_open_price_when_open_pos
    next_step_position = next_step_position.fill_null(0)  # 填充空值
    
    # 计算仓位差异
    diff_position = cur_position - next_step_position
    
    # 计算现金变化
    cash_change = sum_row(diff_position * all_open_price_when_open_pos)
    
    # 计算交易成本
    abs_diff_position = diff_position.select(
        [pl.col(column).sum().abs() for column in diff_position.columns]
    )
    abs_diff_trading_value = sum_row(abs_diff_position * all_open_price_when_open_pos)
    commission_cost = abs_diff_trading_value * commission
    
    logger.debug(f"diff_trading_value: {abs_diff_trading_value}")
    return next_step_position, diff_position, cash_change, commission_cost

def update_positions_and_cash(
    all_symbols: list,
    long_symbol_list: list,
    short_symbol_list: list,
    cur_position: pl.DataFrame,
    all_open_price_when_open_pos: pl.DataFrame,
    total_long_pos_value: float,
    total_short_pos_value: float,
    long_symbol_pos_ratio_dict: Dict,
    short_symbol_pos_ratio_dict: Dict,
    cash: float,
    commission: float,
    cur_close_time: np.datetime64,
    logger
) -> Tuple[pl.DataFrame, pl.DataFrame, float, list, list]:
    """
    更新仓位和现金
    """
    logger.debug("begin to update pos: =========")
    
    # 计算下一步市场价值
    next_step_value = calculate_next_step_market_value(
        all_symbols,
        long_symbol_list,
        short_symbol_list,
        total_long_pos_value,
        total_short_pos_value,
        long_symbol_pos_ratio_dict,
        short_symbol_pos_ratio_dict,
        logger
    )
    
    # 记录开仓价格
    cur_pos_open_price = all_open_price_when_open_pos.clone()
    logger.debug(f"all_open_price_when_open_pos: {all_open_price_when_open_pos}")
    
    # 计算仓位变动和现金变化
    next_step_position, diff_position, cash_change, commission_cost = calculate_position_changes(
        cur_position,
        next_step_value,
        all_open_price_when_open_pos,
        commission,
        logger
    )
    
    # 更新现金
    cash += cash_change
    cash -= commission_cost
    
    logger.debug(f"update position: {cur_close_time} === diff_trading_value: {commission_cost/commission} == cash: {cash}")
    
    return next_step_position, cur_pos_open_price, cash, long_symbol_list, short_symbol_list

def read_delist_info(config: BacktestConfig, logger) -> pl.DataFrame:
    """Read CSV and convert string dates to datetime."""
    if not os.path.exists(config.delist_info_path):
        return pl.DataFrame([])
    df = pl.read_csv(config.delist_info_path).with_columns(
        [
            pl.col("announce_time").str.strptime(
                pl.Datetime("ms", "UTC"), "%Y,%m,%d,%H,%M"
            ),
            pl.col("end_trade_time").str.strptime(
                pl.Datetime("ms", "UTC"), "%Y,%m,%d,%H,%M"
            ),
        ]
    )
    
    if df.shape[0] == 0:
        logger.warning(f"Empty delist info dataframe: {config.delist_info_path}")

    return df

def GetSinglePnL(
    logger,
    result_hour,
    FACTOR_NAME,
    config: BacktestConfig,
    group_num=20,
    long_factor_combination_list=[1, 2, 3],
):
    commission = config.commission
    start_cash = config.start_cash
    eps = config.eps
    use_hour_data = config.use_hour_data
    enable_early_cut_loss = config.enable_early_cut_loss
    cut_loss_pct_threshold = config.cut_loss_pct_threshold
    vol_filter_ratio = config.vol_filter_ratio
    long_trade_rank_ratio = config.long_trade_rank_ratio
    short_trade_rank_ratio = config.short_trade_rank_ratio
    pos_ret_threshold = config.pos_ret_threshold
    debug = config.debug
    use_factor_pos_ratio = config.use_factor_pos_ratio
    overall_leverage = config.leverage

    trade_with_rank = config.trade_with_rank
    update_position_time = config.update_position_time
    delist_info = read_delist_info(config, logger)
    

    logger.info(f"start get SinglePnl: {result_hour}")

    # Prepare data
    future_n_day_open_ret, long_scale, factors, open_df, bar_close_vol = prepare_data(
        result_hour, FACTOR_NAME, update_position_time
    )

    # --------------- Initialize variables ---------------
    # 在每一根k线走完的时候计算因子值，然后调仓
    # 所以要遍历每一个close_time，进行调仓
    time_array = factors["close_time"].to_numpy()[:-1]   # the last line doesn't have next day return

    columns = future_n_day_open_ret.columns[1:]  # means all symbols
    all_symbols = columns
    today_pnl = 1
    pnl_records: List[PnLEntry] = []
    cash = start_cash
    latest_pnl = start_cash
    cur_position = pl.DataFrame(
        {col: [value] for col, value in zip(columns, [0] * len(columns))}
    )
    next_step_position = pl.DataFrame(
        {col: [value] for col, value in zip(columns, [0] * len(columns))}
    )

    # 用日线数据的时候，每一行是一天；小时数据的时候，每24行是一天
    update_row_index = (
        update_position_time if use_hour_data == 0 else update_position_time * 24
    )

    # used to count the PnL for current pos & stop loss early
    cur_pos_open_price: pl.DataFrame = None
    cur_hold_long_symbol_list = []
    cur_hold_short_symbol_list = []
    bar_open_time_to_close_pos = time_array[0] # only for init, no meaning

    for i, cur_close_time in enumerate(time_array):
        # 用下一根k线的开盘价作为调仓价格，open_time是1ms后
        time_delta: np.timedelta64 = np.timedelta64(1, "ms")

        next_bar_open_time = cur_close_time + time_delta
        all_open_price_when_open_pos = open_df.filter(
            pl.col("open_time") == next_bar_open_time
        ).drop("open_time")

        # check early cut loss
        if enable_early_cut_loss and cur_pos_open_price is not None:
            cur_position, cash = handle_early_cut_loss(
                cur_position, cur_pos_open_price, all_open_price_when_open_pos,
                cur_hold_long_symbol_list, cur_hold_short_symbol_list,
                cash, cut_loss_pct_threshold, commission, eps, logger
            )

        # Handle delisted symbols
        cur_position, cash = handle_delisted_symbols(
            cur_position, delist_info, cur_close_time,
            bar_open_time_to_close_pos, all_open_price_when_open_pos,
            cash, logger
        )

        if i % update_row_index == 0:
            # for debug and check pnl
            # 计算 {update_position_time} 天后的平仓时间(也是在开盘价平仓，所以是open_time)
            # 每次调仓才会更新这个数值，表示什么时候关闭当前的仓位
            bar_open_time_to_close_pos = next_bar_open_time + np.timedelta64(
                update_position_time, "D"
            )

            logger.debug(
                f"cur time: {cur_close_time} == next bar open time: {next_bar_open_time} == close pos time: {bar_open_time_to_close_pos}"
            )

            # Before update the position, calc the total cash+position value

            # 选取当前时间点的数据
            current_factors = factors.filter(
                pl.col("close_time") == cur_close_time
            ).drop("close_time")

            tradeable_symbol_list, _ = GetTradeableSymbolList(
                current_factors,
                next_bar_open_time,
                delist_info= delist_info
            )

            # Filter symbols based on volume
            cur_bar_close_vol = bar_close_vol.filter(
                pl.col("close_time") == cur_close_time
            )
            tradeable_symbol_list = TradeableSymbolFilter(
                tradeable_symbol_list,
                cur_bar_close_vol,
                vol_filter_ratio,
                cur_close_time,
                logger
            )

            logger.debug (f'tradeable list {tradeable_symbol_list} == next bar open time: {next_bar_open_time}')
            # -------------------------

            if trade_with_rank == 0 and len(tradeable_symbol_list) < group_num:
                # 如果可以交易的symbol数目小于组数，那么无法进行交易
                pnl_records.append(PnLEntry(next_bar_open_time, today_pnl))
                continue

            # 准备和处理因子数据
            sorted_long_factors = prepare_factor_data(current_factors, tradeable_symbol_list)
            
            if len(tradeable_symbol_list) == 0:
                pnl_records.append(PnLEntry(next_bar_open_time, latest_pnl))
                logger.warning(f"no tradeable symbol at {cur_close_time}")
                continue
            
            # 计算分组大小
            group_size = max(int(sorted_long_factors.height / group_num), 1)
            
            # 创建排名或分组标签
            sorted_long_factors = create_rank_labels(sorted_long_factors, trade_with_rank, group_size)
            
            logger.debug(f"sorted_long_factors: {sorted_long_factors}")
            
            # 计算当前交易数量
            cur_trade_num = min(abs(trade_with_rank), sorted_long_factors.height / 2) if trade_with_rank != 0 else 0
            
            # 选择交易股票
            long_symbol_list, short_symbol_list = select_trading_stocks(
                sorted_long_factors,
                trade_with_rank,
                cur_trade_num,
                long_trade_rank_ratio,
                short_trade_rank_ratio,
                group_num,
                long_factor_combination_list
            )

            # 计算仓位比例
            total_long_pos_value, total_short_pos_value, long_symbol_pos_ratio_dict, short_symbol_pos_ratio_dict = \
                calculate_position_ratios(latest_pnl, overall_leverage, long_symbol_list, short_symbol_list, cur_close_time, logger)
            
            if debug:
                # 计算因子平均值
                mean_expected_ret = sorted_long_factors["factor_value"].drop_nans().drop_nulls().mean()
                logger.debug(f"mean expected ret (mean factor): {mean_expected_ret}")

                # 分析和记录交易详情
                long_symbol_pos_ratio_dict, short_symbol_pos_ratio_dict = analyze_and_log_trading_details(
                    cur_position,
                    next_step_position,
                    cur_close_time,
                    next_bar_open_time,
                    long_symbol_list,
                    short_symbol_list,
                    future_n_day_open_ret,
                    all_open_price_when_open_pos,
                    current_factors,
                    use_factor_pos_ratio,
                    long_symbol_pos_ratio_dict,
                    short_symbol_pos_ratio_dict,
                    eps,
                    logger
                )

            # 更新仓位和现金
            next_step_position, cur_pos_open_price, cash, cur_hold_long_symbol_list, cur_hold_short_symbol_list = \
                update_positions_and_cash(
                    all_symbols, long_symbol_list, short_symbol_list, cur_position,
                    all_open_price_when_open_pos, total_long_pos_value, total_short_pos_value,
                    long_symbol_pos_ratio_dict, short_symbol_pos_ratio_dict,
                    cash, commission, cur_close_time, logger
                )
            
            cur_position = next_step_position.clone()  # 完成调仓

        latest_pnl = cash + sum_row(all_open_price_when_open_pos * cur_position)

        if len(pnl_records) > 0:
            logger.info(
                f"PnL: {cur_close_time} {latest_pnl} === {(latest_pnl / pnl_records[-1].value - 1) * 100:.3f}%"
            )
        pnl_records.append(PnLEntry(next_bar_open_time, latest_pnl))
    return pnl_records, AnalysePnLTrace([x.value for x in pnl_records]), " "
