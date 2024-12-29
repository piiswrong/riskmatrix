import polars as pl
from datetime import datetime
from typing import Tuple, List
import warnings
import sys
import numpy as np
import scipy
import pandas as pd
import statsmodels.api as sm


def analyze_inf_values(data, factor_name="alpha40"):
    """
    Analyze infinite values in the factor using list conversion
    """
    factor_list = data[factor_name].to_list()

    # Count infinities using list
    inf_mask = np.isinf(factor_list)
    pos_inf = np.sum(np.logical_and(inf_mask, np.array(factor_list) > 0))
    neg_inf = np.sum(np.logical_and(inf_mask, np.array(factor_list) < 0))

    print(f"\n=== Infinity Analysis for {factor_name} ===")
    print(f"Total values: {len(factor_list)}")
    print(f"Positive infinities: {pos_inf} ({pos_inf/len(factor_list)*100:.4f}%)")
    print(f"Negative infinities: {neg_inf} ({neg_inf/len(factor_list)*100:.4f}%)")

    # Get indices of infinite values
    inf_indices = np.where(inf_mask)[0]

    print("\nSample of values around infinities:")
    for idx in inf_indices[:5]:  # Look at first 5 infinite values
        start_idx = max(0, idx - 2)
        end_idx = min(len(factor_list), idx + 3)
        window = factor_list[start_idx:end_idx]
        print(f"\nAround index {idx}:")
        print(window)

    # Get non-infinite range
    valid_data = [x for x in factor_list if not np.isinf(x)]
    print(f"\nRange excluding infinities:")
    print(f"Min: {min(valid_data):.6f}")
    print(f"Max: {max(valid_data):.6f}")


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
    # date_threshold: datetime = datetime(2023, 1, 1),
    date_threshold: datetime = datetime(2024, 8, 1),  # temp for small_hour_data
) -> pl.DataFrame:

    factor_num = len(factor_combination_list)

    for cur_update_position_time in range(1, day_num + 1):
        cur_fut_ret_column_name = f"close_price_fut_{cur_update_position_time}day_ret"

        non_nan_result = input_df.filter(
            (pl.col(cur_fut_ret_column_name).is_not_nan())
            & (pl.col(cur_fut_ret_column_name).is_not_null())
        ).sort(["open_time", "symbol"])

        non_nan_linear_x = non_nan_result.select(
            ["open_time", "symbol", cur_fut_ret_column_name] + factor_combination_list
        )

        date_threshold_ms = date_threshold.timestamp() * 1000
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


if __name__ == "__main__":
    path = "small_hour_factor.parquet"
    path = "../production/hour_factor.parquet"
    data = pl.read_parquet(path)

    # deal with inf temp
    columns_to_clean = ["alpha40"]
    # data = data.with_columns(
    #     pl.col(columns_to_clean)
    #     .fill_nan(0)
    #     .fill_infinite(0)
    # )

    # data = data.with_columns(
    #     pl.col("alpha40")
    #     .fill_nan(0)
    #     .replace_inf(0, -0)  # first value replaces inf, second replaces -inf
    #     .alias("alpha40")
    # )

    data = data.with_columns(
        pl.when(pl.col("alpha40").is_infinite())
        .then(0)
        .otherwise(pl.col("alpha40"))
        .fill_nan(0)
        .alias("alpha40_clean")
    )

    FACTOR_COMBINATION_LIST = [
        "amihud",
        "alpha30",
        "alpha36",
        "alpha45",
        # "alpha40",
        "alpha40_clean",
    ]
    UPDATE_POSITION_TIME = 10

    data = CalcLinearCompoundFactor(data, UPDATE_POSITION_TIME, FACTOR_COMBINATION_LIST)

    print(f"result: ", data)
    data.write_parquet("data/small_hour_factor.parquet")
