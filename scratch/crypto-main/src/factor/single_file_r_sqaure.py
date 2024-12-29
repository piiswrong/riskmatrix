import polars as pl
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import seaborn as sns
from datetime import datetime


def calculate_factor_stats(df: pl.DataFrame, y_col: str, x_col: str) -> Dict:
    """
    计算单因子的统计指标

    Args:
        df: 输入数据框
        y_col: 预测目标列名
        x_col: 因子列名

    Returns:
        包含各种统计指标的字典
    """
    # 转换为numpy数组进行计算
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # 移除含有空值的行
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return None

    # 计算峰度和偏度
    kurtosis = stats.kurtosis(x)
    skewness = stats.skew(x)

    # 计算相关系数
    correlation, p_value = stats.pearsonr(x, y)

    # 计算IC
    rank_ic, rank_p_value = stats.spearmanr(x, y)

    # 简单线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # R方
    r_squared = r2_score(y, slope * x + intercept)

    # 计算MSE
    y_pred = slope * x + intercept
    mse = mean_squared_error(y, y_pred)

    # 计算信息比率 (IR)
    ic_series = []
    for date in df["open_time"].unique():
        date_df = df.filter(pl.col("open_time") == date)
        if len(date_df) > 2:  # 确保有足够的数据计算相关性
            date_ic, _ = stats.spearmanr(date_df[x_col], date_df[y_col])
            if not np.isnan(date_ic):
                ic_series.append(date_ic)

    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    ir = ic_mean / ic_std if ic_std != 0 else 0

    return {
        "correlation": correlation,
        "p_value": p_value,
        "rank_ic": rank_ic,
        "rank_ic_p_value": rank_p_value,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "mse": mse,
        "information_ratio": ir,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "sample_size": len(x),
        "kurtosis": kurtosis,
        "skewness": skewness,
    }


def print_factor_report(stats: Dict) -> None:
    """
    打印因子分析报告
    """
    print("\n=== 单因子分析报告 ===")
    print(f"样本量: {stats['sample_size']}")
    print(f"\n分布特征:")
    print(f"峰度: {stats['kurtosis']:.4f}")
    print(f"偏度: {stats['skewness']:.4f}")
    print(f"\n相关性分析:")
    print(
        f"Pearson相关系数: {stats['correlation']:.4f} (p-value: {stats['p_value']:.4f})"
    )
    print(f"Rank IC: {stats['rank_ic']:.4f} (p-value: {stats['rank_ic_p_value']:.4f})")
    print(f"\n回归分析:")
    print(f"斜率: {stats['slope']:.4f}")
    print(f"截距: {stats['intercept']:.4f}")
    print(f"R平方: {stats['r_squared']:.4f}")
    print(f"均方误差(MSE): {stats['mse']:.4f}")
    print(f"\nIC分析:")
    print(f"IC均值: {stats['ic_mean']:.4f}")
    print(f"IC标准差: {stats['ic_std']:.4f}")
    print(f"信息比率(IR): {stats['information_ratio']:.4f}")


def plot_factor_analysis(df: pl.DataFrame, y_col: str, x_col: str, stats: Dict) -> None:
    """
    绘制因子分析图表
    """
    # plt.figure(figsize=(15, 10))
    plt.figure(figsize=(10, 5))

    # 散点图
    plt.subplot(2, 2, 1)
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot (R² = {stats["r_squared"]:.4f})')

    # 添加回归线
    x = df[x_col].to_numpy()
    y_pred = stats["slope"] * x + stats["intercept"]
    plt.plot(x, y_pred, color="red", label="Regression Line")
    plt.legend()

    # IC时间序列图
    plt.subplot(2, 2, 2)
    ic_series = []
    dates = []
    # TODO
    # for date in df["open_time"].unique():
    #     date_df = df.filter(pl.col("open_time") == date)
    #     if len(date_df) > 2:
    #         ic, _ = stats.spearmanr(date_df[x_col], date_df[y_col])
    #         if not np.isnan(ic):
    #             ic_series.append(ic)
    #             dates.append(date)

    plt.plot(dates, ic_series)
    plt.xlabel("Date")
    plt.ylabel("Rank IC")
    plt.title(f'Rank IC Time Series (Mean = {stats["ic_mean"]:.4f})')
    plt.xticks(rotation=45)

    # 因子收益分布图
    plt.subplot(2, 2, 3)
    returns = df[y_col].to_numpy()
    plt.hist(returns[~np.isnan(returns)], bins=50)
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("Return Distribution")

    # 因子暴露分布图
    plt.subplot(2, 2, 4)
    exposures = df[x_col].to_numpy()
    plt.hist(exposures[~np.isnan(exposures)], bins=50)
    plt.xlabel("Factor Exposure")
    plt.ylabel("Frequency")
    plt.title("Factor Exposure Distribution")

    plt.tight_layout()
    plt.show()


def analyze_single_factor(
    df: pl.DataFrame, predict_y_col: str, x_col: str
) -> Tuple[Dict, None]:
    """
    进行单因子分析的主函数

    Args:
        df: 输入数据框
        predict_y_col: 预测目标列名
        x_col: 因子列名

    Returns:
        统计指标字典和图表
    """
    # 计算统计指标
    stats = calculate_factor_stats(df, predict_y_col, x_col)

    if stats is None:
        print("数据不足，无法进行分析")
        return None

    # 打印报告
    print_factor_report(stats)

    # 绘制分析图表
    plot_factor_analysis(df, predict_y_col, x_col, stats)

    return stats


def main():
    # 读取数据
    DATA_PATH = "data/hour_factor.parquet"
    DATA_PATH = "production/all_data_1d_boris_converted_compound_prod.parquet"
    df = pl.read_parquet(DATA_PATH)
    print(f"df col: {df.columns} {df}")

    # df = df.filter(pl.col("symbol") == "ETHUSDT")
    print("df: ", df)

    # 设置参数
    day_num = 1
    PREDICT_Y_COL = f"close_price_fut_{day_num}day_ret"
    X_COL = "past_1day_close_return"
    X_COL = "alpha30"
    X_COL = "amihud"
    X_COL = "alpha40"
    X_COL = f"linear_compound_factor_{day_num}day"

    # 进行单因子分析
    print(f"single factor: {X_COL}  == predict col: {PREDICT_Y_COL}")
    stats = analyze_single_factor(df, PREDICT_Y_COL, X_COL)


if __name__ == "__main__":
    main()
