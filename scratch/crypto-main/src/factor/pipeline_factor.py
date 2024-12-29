import polars as pl
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import numpy as np
import statsmodels.api as sm
from crypto_utils.factor_utils import sign
from factor.alpha101 import *
import pandas as pd

POS_RET_PCT_SCALE_THRESHOLD = 0.01


def ConvertPdWideToLong(result_df, factor_name):
    # Convert the result DataFrame to a wide format
    result_wide = result_df.unstack()
    assert isinstance(
        result_wide, pd.Series
    ), f"should be pd.series after unstack, but got {type(result_wide)}"
    result_wide.name = factor_name

    # print (f'after unstack: {result_wide.shape} == {result_wide}')

    result_wide = result_wide.to_frame().reset_index()
    assert (
        type(result_wide) is pd.DataFrame
    ), f"should be pd.DataFrame after reset_index, but got {type(result_wide)}"

    # Convert to Polars DataFrame
    pl_result_wide = pl.from_pandas(result_wide)

    pl_result_long = pl_result_wide

    # If there are any datetime(ns) columns, convert them to datetime(ms)
    datetime_ns_columns = [
        col
        for col, dtype in pl_result_long.schema.items()
        if dtype == pl.Datetime("ns")
    ]
    if datetime_ns_columns:
        for col in datetime_ns_columns:
            pl_result_long = pl_result_long.with_columns(
                pl_result_long[col].cast(pl.Datetime("ms"))
            )

    return pl_result_long


def CompareWithSmallRangeResult(
    full_time_data: pd.DataFrame, small_range_data: pd.DataFrame
):
    # check if the full_time_data and small_range_data are the same, in these days
    date_array = [
        pl.datetime(2023, 6, 1),
        pl.datetime(2024, 1, 1),
        pl.datetime(2024, 6, 1),
    ]

    for check_date in date_array:
        print(f" ===== check on date {check_date}")
        select_large_data = full_time_data.filter(pl.col("open_time") == check_date)
        select_small_data = small_range_data.filter(pl.col("open_time") == check_date)

        compare_df = select_large_data == select_small_data
        print(
            " ======= begin to check if any factor use future information ============="
        )
        for i in compare_df.columns:
            # for each columns, should all be True (equal)
            # otherwise, our factor use future information
            assert compare_df[
                i
            ].all(), f"date {check_date} ==column {i} differ === {compare_df[i]} \n full data: {select_large_data} small_data: {select_small_data}"
            print(f"   factor {i} check done.")


def AddPastReturnFactor(input_df: pl.DataFrame, day_num: int) -> pl.DataFrame:
    input_df = input_df.sort(["symbol", "open_time"])
    # 千万注意，这里是计算此时刻相对于前一时刻的return，不能使用未来信息 -> shift(1)是整体往下移动
    for i in range(1, day_num + 1):
        input_df = (
            input_df.join(
                input_df.select(
                    pl.col("open_datetime").dt.offset_by(f"{i}d"),  # past i day
                    pl.col("close").alias(f"close_past_{i}day"),
                    "symbol",
                ),
                on=["open_datetime", "symbol"],
                how="left",
            )
            .with_columns(
                ((pl.col("close") / pl.col(f"close_past_{i}day") - 1) * 100).alias(
                    f"past_{i}day_close_return"
                )
            )
            .drop(f"close_past_{i}day")
        )

    # 默认使用过去1天的return作为return列
    input_df = input_df.with_columns(pl.col("past_1day_close_return").alias("return"))

    return input_df


def CalcDayPositionScale(
    input_df: pl.DataFrame, day_num: int, trade_long_rank: int, trade_short_rank: int
) -> pl.DataFrame:
    agg_avg_ret_list = []
    for i in range(1, day_num + 1):
        for side in ["long", "short"]:
            sort_desc = True if side == "long" else False
            trade_rank_num = trade_long_rank if side == "long" else trade_short_rank

            input_df = input_df.with_columns(
                pl.col(f"linear_compound_factor_{i}day")
                .rank(descending=sort_desc)
                .over("open_time")
                .alias(f"symbol_rank_{side}_{i}day")
            ).with_columns(
                pl.when(pl.col(f"symbol_rank_{side}_{i}day") <= trade_rank_num)
                .then(pl.col(f"close_price_fut_{i}day_ret"))
                .otherwise(None)
                .alias(f"total_{side}_value_scale_{i}day")
            )

            cur_agg_avg_ret = (
                input_df.group_by("open_time")
                .agg(
                    pl.col(f"total_{side}_value_scale_{i}day")
                    .mean()
                    .alias(f"fut_mean_{side}_ret_{i}day"),
                )
                .sort("open_time")
            )
            agg_avg_ret_list.append(cur_agg_avg_ret)

    agg_avg_ret_df = agg_avg_ret_list[0]
    for df in agg_avg_ret_list[1:]:
        agg_avg_ret_df = agg_avg_ret_df.join(df, on="open_time", how="right")

    # Sort the final DataFrame by open_time
    agg_avg_ret_df = agg_avg_ret_df.sort("open_time")
    agg_avg_ret_df = agg_avg_ret_df.select(
        pl.col(
            ["open_time"]
            + [col for col in agg_avg_ret_df.columns if col != "open_time"]
        )
    )

    for i in range(1, day_num + 1):
        for side in ["long", "short"]:
            agg_avg_ret_df = agg_avg_ret_df.with_columns(
                pl.col(f"fut_mean_{side}_ret_{i}day")
                .shift(i)
                .alias(f"past_mean_{side}_ret_{i}day"),
            )

    for i in range(1, day_num + 1):
        for side in ["long", "short"]:
            bullish_scale = 1.2 if side == "long" else 0.8
            bearish_scale = 0.8 if side == "long" else 1.2
            agg_avg_ret_df = agg_avg_ret_df.with_columns(
                pl.when(
                    (pl.col(f"past_mean_long_ret_{i}day") > POS_RET_PCT_SCALE_THRESHOLD)
                    & (
                        pl.col(f"past_mean_short_ret_{i}day")
                        > POS_RET_PCT_SCALE_THRESHOLD
                    )
                )
                .then(bullish_scale)
                .when(
                    (pl.col(f"past_mean_long_ret_{i}day") < POS_RET_PCT_SCALE_THRESHOLD)
                    & (
                        pl.col(f"past_mean_short_ret_{i}day")
                        < POS_RET_PCT_SCALE_THRESHOLD
                    )
                )
                .then(bearish_scale)
                .otherwise(1.0)
                .alias(f"{side}_value_scale_{i}day")
            )
    return input_df, agg_avg_ret_df


def AddTotalPosValueScale(
    input_df: pl.DataFrame, day_num: int, trade_long_rank: int, trade_short_rank: int
) -> pl.DataFrame:
    _, day_scale_df = CalcDayPositionScale(
        input_df, day_num, trade_long_rank, trade_short_rank
    )

    # only need the scale columns
    select_col = [col for col in day_scale_df.columns if "scale" in col]
    day_scale_df = day_scale_df.select(pl.col(["open_time"] + select_col))

    input_df = input_df.join(day_scale_df, on="open_time", how="left")

    return input_df, day_scale_df


def AddAmihud(input_df: pl.DataFrame, window_size: int = 10) -> pl.DataFrame:
    # Calculate rolling sums for absolute returns and quote volume
    input_df = input_df.with_columns(
        pl.col("return")
        .abs()
        .rolling_sum(window_size=window_size)
        .over("symbol")
        .alias("rolling_abs_return_sum"),
        pl.col("quote_volume")
        .rolling_sum(window_size=window_size)
        .over("symbol")
        .alias("rolling_quote_volume_sum"),
    )

    # Calculate Amihud illiquidity measure
    input_df = input_df.with_columns(
        (pl.col("rolling_abs_return_sum") / pl.col("rolling_quote_volume_sum"))
        .over("symbol")  # Apply the final operation within each symbol group
        .alias("amihud")
    )

    # Drop intermediate columns
    input_df = input_df.drop(["rolling_abs_return_sum", "rolling_quote_volume_sum"])

    return input_df


def AddReturnAutocorr(
    input_df: pl.DataFrame, window_size: int = 20, lag: int = 1
) -> pl.DataFrame:
    # Calculate rolling sums for absolute returns and quote volume
    input_df = input_df.with_columns(
        pl.rolling_corr(
            pl.col("return"), pl.col("return").shift(1), window_size=window_size
        )
        .over("symbol")
        .alias("return_autocorr_" + str(lag))
    )
    return input_df


def AddReturnSkewness(input_df: pl.DataFrame, window_size: int = 20) -> pl.DataFrame:
    # Calculate rolling sums for absolute returns and quote volume
    input_df = input_df.with_columns(
        pl.col("return")
        .rolling_skew(window_size)
        .over("symbol")
        .alias("return_skewness")
    )
    return input_df


def AddFutureRetCol(input_df: pl.DataFrame, day_num: int) -> pl.DataFrame:
    input_df = input_df.sort(["symbol", "open_time"])
    for i in range(1, day_num + 1):
        input_df = (
            input_df.join(
                input_df.select(
                    pl.col("open_datetime").dt.offset_by(f"-{i}d"),  # future i day
                    pl.col("open").alias(f"open_fut_{i}day"),
                    pl.col("close").alias(f"close_fut_{i}day"),
                    "symbol",
                ),
                on=["open_datetime", "symbol"],
                how="left",
            )
            .with_columns(
                ((pl.col(f"open_fut_{i}day") / pl.col("open") - 1) * 100).alias(
                    f"open_price_fut_{i}day_ret"
                ),
                ((pl.col(f"close_fut_{i}day") / pl.col("close") - 1) * 100).alias(
                    f"close_price_fut_{i}day_ret"
                ),
            )
            .drop(f"open_fut_{i}day", f"close_fut_{i}day")
        )

    return input_df


def fama_macbeth_get_factor_weight(
    train_data: pl.DataFrame,
    update_pos_days: int,
    factor_num: int,
    factor_combination_list: List[str],
) -> Tuple[np.ndarray, float]:
    # Drop rows containing any null values
    train_data = train_data.drop_nulls()

    total_weights_sum = np.zeros(factor_num)
    unique_times = train_data.select(pl.col("open_time").sort()).unique().to_numpy()

    constant_sum = 0.0

    for each_time in unique_times:
        y_column_name = f"close_price_fut_{update_pos_days}day_ret"
        assert (
            y_column_name in train_data.columns
        ), f"Column {y_column_name} (as y) not found in train data"

        slice_data = train_data.filter(pl.col("open_time") == each_time).fill_nan(0)

        X = slice_data[factor_combination_list].to_numpy()
        X = sm.add_constant(X)  # Add constant term (intercept)
        y = slice_data[y_column_name].to_numpy()

        model = sm.OLS(y, X)
        results = model.fit()
        weights = results.params[1:]
        constant_sum += results.params[0]  # constant term

        while weights.shape[0] < total_weights_sum.shape[0]:
            weights = np.append(weights, 0)

        total_weights_sum += weights

    total_weights_sum /= len(unique_times)
    avg_const_term = constant_sum / len(unique_times)

    return total_weights_sum, avg_const_term


def CalcLinearCompoundFactor(
    input_df: pl.DataFrame,
    day_num: int,
    factor_combination_list: list,
    date_threshold: datetime = datetime(2023, 1, 1),
    training_day_interval: Optional[int] = None,  # 默认为None，表示使用全部历史数据
) -> pl.DataFrame:
    print(f"fit linear model, date_threshold: {date_threshold}")
    if training_day_interval is not None:
        print(f"using training interval: {training_day_interval} days")

    factor_num = len(factor_combination_list)
    date_threshold_ms = date_threshold.timestamp() * 1000

    for cur_update_position_time in range(1, day_num + 1):
        cur_fut_ret_column_name = f"close_price_fut_{cur_update_position_time}day_ret"
        non_nan_result = input_df.filter(
            (pl.col(cur_fut_ret_column_name).is_not_nan())
            & (pl.col(cur_fut_ret_column_name).is_not_null())
        ).sort(["open_time", "symbol"])

        non_nan_linear_x = non_nan_result.select(
            ["open_time", "symbol", cur_fut_ret_column_name] + factor_combination_list
        )

        # 根据是否设置了training_day_interval来决定使用多少训练数据
        if training_day_interval is not None:
            training_start_date = date_threshold - timedelta(days=training_day_interval)
            training_start_ms = training_start_date.timestamp() * 1000
            linear_x_train = non_nan_linear_x.filter(
                (pl.col("open_time") >= training_start_ms)
                & (pl.col("open_time") < date_threshold_ms)
            )
        else:
            # 使用全部历史数据
            linear_x_train = non_nan_linear_x.filter(
                pl.col("open_time") < date_threshold_ms
            )

        weighted_factors, const_term = fama_macbeth_get_factor_weight(
            linear_x_train,
            cur_update_position_time,
            factor_num=factor_num,
            factor_combination_list=factor_combination_list,
        )

        weighted_sum_expr = pl.lit(const_term)
        for factor, weight in zip(factor_combination_list, weighted_factors):
            weighted_sum_expr += pl.col(factor) * weight

        # 这个代码用于实际交易的时候，我们只需要使用权重计算未来收益率
        input_df = input_df.with_columns(
            weighted_sum_expr.alias(
                f"linear_compound_factor_{cur_update_position_time}day"
            )
        )

    return input_df.filter(
        pl.col("open_time") >= date_threshold_ms
    )  # only return the data after the threshold


def CalcRollingLinearCompoundFactor(
    input_df: pl.DataFrame,
    day_num: int,
    factor_combination_list: list,
    threshold_dates: List[datetime],
    training_day_interval: Optional[int] = None,
) -> pl.DataFrame:
    """
    Calculate linear compound factors using rolling time windows defined by threshold dates.
    Reuses CalcLinearCompoundFactor for each window calculation.

    Args:
        input_df: Input DataFrame
        day_num: Number of future days to calculate returns for
        factor_combination_list: List of factor columns to use
        threshold_dates: List of dates defining the rolling windows, must be sorted ascending
        training_day_interval: Optional training window length in days

    Returns:
        DataFrame with calculated linear compound factors for dates after first threshold
    """
    print(f"fit rolling linear model with {len(threshold_dates)} threshold dates")
    result_dfs = []

    # Process each time window
    for i in range(len(threshold_dates) - 1):
        current_threshold = threshold_dates[i]
        next_threshold = threshold_dates[i + 1]
        print(f"Processing window {i+1}: {current_threshold} to {next_threshold}")

        # Calculate factors using existing function
        window_result = CalcLinearCompoundFactor(
            input_df=input_df,
            day_num=day_num,
            factor_combination_list=factor_combination_list,
            date_threshold=current_threshold,
            training_day_interval=training_day_interval,
        )

        # Filter for current window only
        next_threshold_ms = next_threshold.timestamp() * 1000
        window_result = window_result.filter(pl.col("open_time") < next_threshold_ms)

        result_dfs.append(window_result)

    # Combine all windows
    return pl.concat(result_dfs).sort(["open_time", "symbol"])


def AddVolatilityCol(input_df: pl.DataFrame) -> pl.DataFrame:
    def calculate_volatility(group, bar_name: str, window_size: int = 30):
        return group.with_columns(
            [
                pl.col(bar_name)
                .pct_change()
                .rolling_std(window_size=window_size)
                .alias(f"{bar_name}_price_volatility")
            ]
        )

    input_df = input_df.group_by("symbol").map_groups(
        lambda x: calculate_volatility(x, "open", window_size=30)
    )
    input_df = input_df.group_by("symbol").map_groups(
        lambda x: calculate_volatility(x, "close", window_size=30)
    )
    return input_df


def AddPastVolMean(input_df: pl.DataFrame, day_num: int) -> pl.DataFrame:
    input_df = input_df.sort(["symbol", "open_time"])

    for i in range(1, day_num + 1):
        input_df = input_df.with_columns(
            [
                pl.col("volume")
                .rolling_mean(window_size=i)
                .over("symbol")
                .alias(f"vol_{i}day_mean"),
                pl.col("quote_volume")
                .rolling_mean(window_size=i)
                .over("symbol")
                .alias(f"quote_vol_{i}day_mean"),
            ]
        )

    return input_df


def AddID(input_df: pl.DataFrame) -> pl.DataFrame:
    input_df = input_df.with_columns(
        (pl.col("close") / pl.col("open") - 1).over("symbol").alias("ret1")
    )
    input_df = input_df.sort(by=["symbol", "open_time"])

    # 计算 ID
    input_df = input_df.with_columns(
        (
            sign(pl.col("ret1"))
            * (
                pl.col("ret1").rolling_max(window_size=48)
                - pl.col("ret1").rolling_min(window_size=72)
            )
            / pl.col("close")
        )
        .over("symbol")
        .alias("ID")
    )
    input_df = input_df.with_columns(pl.col("ID").clip(-0.2, 0.3).alias("ID"))
    input_df = input_df.drop("ret1")
    return input_df


def normalize_factors(
    input_data: pl.DataFrame, factor_combination_list: list
) -> pl.DataFrame:
    """
    Normalize factors by subtracting mean and dividing by standard deviation.

    Args:
        input_data: Input DataFrame containing factors to normalize
        factor_combination_list: List of factor column names to normalize

    Returns:
        DataFrame with normalized factors
    """
    input_data = input_data.sort(by=["symbol", "open_time"])
    original_columns = input_data.columns

    for c in factor_combination_list:
        print(f"normalizing column: {c}")
        input_data = input_data.with_columns(
            pl.col(c).mean().over("open_time").alias("mean_" + c)
        )
        input_data = input_data.with_columns(
            pl.col(c).std().over("open_time").alias("std_" + c)
        )
        input_data = input_data.with_columns(
            ((pl.col(c) - pl.col("mean_" + c)) / pl.col("std_" + c)).alias(c)
        )

    input_data = input_data.select(original_columns)
    return input_data


def pipeline_main(
    input_path,
    output_path,
    factor_combination_list: List[str],
    save_options: str,
    date: Optional[str]=None,
    backtest_mode=False,
    rolling_day=None,
):
    print(f"pipeline_main (), backtest_mode {backtest_mode}")

    MAX_UPDATE_POSITION_TIME = 10

    input_data = (
        pl.read_parquet(input_path)
        .sort(["open_time", "symbol"])
        .filter(pl.col("volume") > 0)
    )
    input_data = input_data.with_columns(
        pl.from_epoch("open_time", time_unit="ms").alias("open_datetime")
    )

    if date is not None:
        # date is a string like "20241125"
        input_data = input_data.filter(
            pl.col("open_datetime")
            <= pl.datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
        ).sort(["open_time", "symbol"])

    exclude_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BCHUSDT",
        "LTCUSDT",
        "ETCUSDT",
        "LINKUSDT",
        "AVAXUSDT",
        "SOLUSDT",
        "DARBTC",
        "IRISBTC",
        "RAREBRL",
        "THETAETH",
        "UTKBTC",
    ]
    input_data = input_data.filter(~pl.col("symbol").is_in(exclude_symbols))
    input_data = input_data.filter(pl.col("symbol").str.ends_with("USDT"))
    input_data = AddPastReturnFactor(input_data, day_num=10)
    input_data = AddVolatilityCol(input_data)

    if "amihud" in factor_combination_list:
        input_data = AddAmihud(input_data)
    if "autocorr" in factor_combination_list:
        input_data = AddReturnAutocorr(input_data, 100, 1)
    if "return_skewness" in factor_combination_list:
        # input_data = AddReturnSkewness(input_data, 12)
        input_data = AddReturnSkewness(input_data, 7)  # from gubo alpha1
    if "ID" in factor_combination_list:
        input_data = AddID(input_data)

    # Calculate alpha101 factor
    alpha101_factor_list = [x for x in factor_combination_list if "alpha" in x]
    if len(alpha101_factor_list) > 0:
        print(f"begin to calc alpha101 factor: {alpha101_factor_list}")
        input_data = CalcAlpha101Factor(
            input_data, calc_factor_list=alpha101_factor_list
        )

    input_data = normalize_factors(input_data, factor_combination_list)

    input_data = AddPastVolMean(input_data, 10)

    # ==== Calculate linear compound factor ====
    print(f"begin to calc linear compound factor: {factor_combination_list}")
    input_data = AddFutureRetCol(input_data, MAX_UPDATE_POSITION_TIME)

    threshold_date = datetime(2023, 1, 1) if backtest_mode else datetime(2024, 10, 1)

    if backtest_mode:
        threshold_dates = [
            datetime(2023, 1, 1),
            datetime(2026, 1, 1),  # the last one must later than the current date
        ]

        input_data = CalcRollingLinearCompoundFactor(
            input_data,
            MAX_UPDATE_POSITION_TIME,
            factor_combination_list,
            threshold_dates=threshold_dates,
            training_day_interval=rolling_day,
        )
    else:
        input_data = CalcLinearCompoundFactor(
            input_data,
            MAX_UPDATE_POSITION_TIME,
            factor_combination_list,
            date_threshold=threshold_date,
            training_day_interval=720,
        )

    input_data, day_scale_df = AddTotalPosValueScale(
        input_data, day_num=10, trade_long_rank=20, trade_short_rank=10
    )

    if save_options == "save_all" or save_options == "save_per_day":
        # for normal run, save all data
        input_data.write_parquet(output_path)
    else:
        # for backtest and research, remove the last few rows and symbols whose min value is larger then 5 USDT
        assert 0, "should not reach here, delete this branch later"

    return input_data, day_scale_df


def AddVwapColumn(pd_result_hour: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'volume' and 'close' (or your chosen price metric) are numerical and not null
    pd_result_hour["volume"] = pd_result_hour["volume"].astype(float).fillna(0)
    pd_result_hour["close"] = pd_result_hour["close"].astype(float).fillna(0)

    # Calculate cumulative price*volume and cumulative volume
    pd_result_hour["cumulative_pv"] = (
        pd_result_hour["close"] * pd_result_hour["volume"]
    ).cumsum()
    pd_result_hour["cumulative_volume"] = pd_result_hour["volume"].cumsum()

    # Calculate VWAP
    pd_result_hour["vwap"] = (
        pd_result_hour["cumulative_pv"] / pd_result_hour["cumulative_volume"]
    )

    # Cleanup if these intermediate columns are not needed
    pd_result_hour.drop(["cumulative_pv", "cumulative_volume"], axis=1, inplace=True)
    return pd_result_hour


def CalcVwapDf(vol_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    vol_df = vol_df.astype(float).fillna(0)
    close_df = close_df.astype(float).fillna(0)

    cumulative_pv: pd.DataFrame = (close_df * vol_df).cumsum()
    cumulative_volume: pd.DataFrame = vol_df.cumsum()

    vwap_ret_df: pd.DataFrame = cumulative_pv / cumulative_volume

    return vwap_ret_df


def CalcAlpha101Factor(
    result_hour: pl.DataFrame, calc_factor_list, window_size=10
) -> pl.DataFrame:
    pd_result_hour: pd.DataFrame = result_hour.to_pandas()

    def prepare_inputs(all_inputs, required_keys):
        return {k: all_inputs[k] for k in required_keys if k in all_inputs}

    input_close: pd.DataFrame = pd_result_hour[["close_time", "symbol", "close"]].pivot(
        index="close_time", columns="symbol", values="close"
    )

    input_volume: pd.DataFrame = pd_result_hour[
        ["close_time", "symbol", "volume"]
    ].pivot(index="close_time", columns="symbol", values="volume")

    input_high: pd.DataFrame = pd_result_hour[["close_time", "symbol", "high"]].pivot(
        index="close_time", columns="symbol", values="high"
    )

    input_open: pd.DataFrame = pd_result_hour[["close_time", "symbol", "open"]].pivot(
        index="close_time", columns="symbol", values="open"
    )

    input_return: pd.DataFrame = pd_result_hour[
        ["close_time", "symbol", "return"]
    ].pivot(index="close_time", columns="symbol", values="return")

    input_vwap: pd.DataFrame = CalcVwapDf(input_volume, input_close)

    input_dict = {
        "idx": pd_result_hour.index,
        "cols": pd_result_hour.columns,
        "low": pd_result_hour["low"],
        "high": input_high,
        "close": input_close,
        "Open": input_open,
        "volume": input_volume,
        "returns": input_return,  # Assuming 'return' is also a column
        "vwap": input_vwap,  # Assuming 'vwap' is available
    }

    full_required_columns = {
        # "alpha1": ["close", "returns"],  # Example, ensure correct keys are here
        # "alpha2": ["Open", "close", "volume"],
        # "alpha3": ["Open", "volume"],
        # "alpha4": ["low"],
        # "alpha5": ["Open", "vwap", "close"],  # Adjust naming if necessary
        # "alpha6": ["Open", "volume"],
        # "alpha7": ["volume", "close"],
        # "alpha8": ["Open", "returns"],
        # "alpha9": ["close"],
        # "alpha10": ["close"],
        # "alpha11": ["vwap", "close", "volume"],
        # "alpha12": ["volume", "close"],
        "alpha13": ["volume", "close"],
        # "alpha14": ["Open", "volume", "returns"],
        "alpha15": ["high", "volume"],
        "alpha16": ["high", "volume"],
        # "alpha17": ["volume", "close"],
        # "alpha18": ["close", "Open"],
        # "alpha19": ["close", "returns"],
        # "alpha20": ["Open", "high", "close", "low"],
        # # "alpha21": ["volume", "close"],
        # "alpha22": ["high", "volume", "close"],
        # # "alpha23": ["high", "close", "idx", "cols"],
        "alpha24": ["close"],
        "alpha25": ["volume", "returns", "vwap", "high", "close"],
        "alpha26": ["volume", "high"],
        # "alpha27": ["volume", "vwap"], # value wrong, TODO, fix it
        "alpha28": ["volume", "high", "low", "close"],
        # "alpha29": ["close", "returns"],
        "alpha30": ["close", "volume"],
        # # "alpha31": ["close", "low", "volume"],
        # "alpha32": ["close", "vwap"]
        "alpha33": ["Open", "close"],
        "alpha34": ["close", "returns"],
        "alpha35": ["volume", "close", "high", "low", "returns"],
        "alpha36": ["Open", "close", "volume", "returns", "vwap"],
        # "alpha37": ["Open", "close"],
        "alpha38": ["close", "Open"],
        # # "alpha39": ["volume", "close", "returns"],  # killed
        "alpha40": ["high", "volume"],
        # "alpha41": ["high", "low", "vwap"],
        # "alpha42": ["vwap", "close"],
        # "alpha43": ["volume", "close"],
        "alpha44": ["high", "volume"],
        "alpha45": ["close", "volume"],
        "alpha46": ["close"],
        "alpha47": ["volume", "close", "high", "vwap"],
        "alpha49": ["close"],
        "alpha50": ["volume", "vwap"],
        "alpha51": ["close"],
        # "alpha52": ["returns", "volume", "low"],
        # "alpha53": ["close", "high", "low"],
        "alpha54": ["Open", "close", "high", "low"],
        "alpha55": ["high", "low", "close", "volume"],
        # # "alpha56": ["returns", "cap"], # No cap data
        # # "alpha57": ["close", "vwap"], # kiled
        # "alpha60": ["close", "high", "low", "volume"],
        # "alpha61": ["volume", "vwap"],
        "alpha62": ["volume", "high", "low", "Open", "vwap"],
        "alpha64": ["high", "low", "Open", "volume", "vwap"],
        # "alpha65": ["volume", "vwap", "Open"],
        # # "alpha66": ["vwap", "low", "Open", "high"],  # TODO, fix it
        # "alpha68": ["volume", "high", "close", "low"],
        "alpha71": ["volume", "close", "low", "Open", "vwap"],
        # "alpha72": ["volume", "high", "low", "vwap"],
        # # "alpha73": ["vwap", "Open", "low"], # error, TODO, fix it
        "alpha74": ["volume", "close", "high", "vwap"],
        # "alpha75": ["volume", "vwap", "low"],
        # "alpha77": ["volume", "high", "low", "vwap"],
        # "alpha78": ["volume", "low", "vwap"],
        "alpha81": ["volume", "vwap"],
        # "alpha83": ["high", "low", "close", "volume", "vwap"],
        # "alpha84": ["vwap", "close"],
        # "alpha85": ["volume", "high", "close", "low"],
        # "alpha86": ["volume", "close", "Open", "vwap"],
        # "alpha88": ["volume", "Open", "low", "high", "close"],
        # "alpha92": ["volume", "high", "low", "close", "Open"],
        "alpha94": ["volume", "vwap"],
        # "alpha95": ["volume", "high", "low", "Open"],
        # "alpha96": ["volume", "vwap", "close"],
        # "alpha98": ["volume", "Open", "vwap"],
        "alpha99": ["volume", "high", "low"],
        "alpha101": ["close", "Open", "high", "low"],
    }

    required_columns = {}
    for x in calc_factor_list:
        if x in full_required_columns:
            required_columns[x] = full_required_columns[x]

    alpha_functions = {}

    for func_name in required_columns:
        alpha_functions[func_name] = globals()[func_name]

    for func_name, func in alpha_functions.items():
        print(func_name)
        filtered_inputs = prepare_inputs(input_dict, required_columns[func_name])

        result = func(**filtered_inputs)

        if isinstance(result, pd.DataFrame):
            pl_result_long = ConvertPdWideToLong(result, func_name)

            # Join with result_hour, add as a new column in result_hour
            result_hour = result_hour.join(
                pl_result_long, on=["close_time", "symbol"], how="left"
            )
        else:
            print(f"Unexpected result type for {func_name}: {type(result)}")

    return result_hour
