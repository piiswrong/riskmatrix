import polars as pl
import numpy as np
import warnings
import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# import matplotlib.pyplot as plt
# from eval_utils import calculate_statistics


def derive_basic_factors(factor_data):
    factor_data = factor_data.with_columns(  # 平均成交价格
        (pl.col("quote_volume") / pl.col("volume")).alias("avg_trade_price"),
        # 平均买单价格
        (pl.col("taker_buy_quote_volume") / pl.col("taker_buy_volume")).alias(
            "avg_taker_buy_price"
        ),
        # 平均卖单价格
        (
            (pl.col("quote_volume") - pl.col("taker_buy_quote_volume"))
            / (pl.col("volume") - pl.col("taker_buy_volume"))
        ).alias("avg_taker_sell_price"),
    )

    factor_data = factor_data.with_columns(
        [
            # 平均买单溢价
            (
                pl.col("avg_taker_buy_price")
                / pl.col("avg_trade_price")
                * pl.lit(100.0)
                - pl.lit(100.0)
            ).alias("avg_taker_buy_premium_pct"),
            # 平均卖单溢价
            (
                pl.col("avg_taker_sell_price")
                / pl.col("avg_trade_price")
                * pl.lit(100.0)
                - pl.lit(100.0)
            ).alias("avg_taker_sell_premium_pct"),
            # 买卖价格差异
            (pl.col("avg_taker_buy_price") / pl.col("avg_taker_sell_price")).alias(
                "taker_buy_sell_price_ratio"
            ),
            # 买单比例
            (pl.col("taker_buy_volume") / pl.col("volume")).alias("taker_buy_ratio"),
            # # trade count相关
            # (pl.col("quote_volume") / pl.col("count")).alias("avg_quote_vol_per_trade"),
            # 价格波动相关
            ((pl.col("high") - pl.col("low")) / pl.col("open")).alias(
                "price_range_ratio"
            ),
            # ((pl.col("close") - pl.col("open")) / pl.col("open")).alias(
            #     "price_change_ratio"
            # ),  # high corr with alpha33
            (pl.col("high") / pl.col("low") - pl.lit(1)).alias("high_low_ratio"),
            # # 价格位置
            # (
            #     (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
            # ).alias("price_position"),
            # 流动性指标
            (
                pl.col("volume") / ((pl.col("high") - pl.col("low")) / pl.col("low"))
            ).alias("liquidity_ratio"),
        ]
    )

    factor_data = factor_data.drop(
        [
            'avg_trade_price', # 这些均价跟OHLCV数据相关性太高，去掉，只需要知道变化率即可
            'avg_taker_buy_price',
            'avg_taker_sell_price',
        ]
    )
    return factor_data

def derive_cross_sectional_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    计算每个交易对在特定时间点的成交量占全市场的比例。

    参数:
    df : pl.DataFrame
        包含多个交易对数据的 DataFrame，应该包含 'symbol', 'open_time', 'volume', 'quote_volume' 列。

    返回:
    pl.DataFrame
        添加了截面因子的 DataFrame。
    """
    # 按时间分组计算总成交量
    total_volume = df.group_by('open_time').agg([
        pl.sum('volume').alias('total_market_volume'),
        pl.sum('quote_volume').alias('total_market_quote_volume')
    ])

    # 将总成交量信息合并回原始数据框
    df_with_totals = df.join(total_volume, on='open_time')

    # 计算每个交易对的成交量占比
    df_with_totals = df_with_totals.with_columns([
        (pl.col('volume') / pl.col('total_market_volume') * pl.lit(100)).alias('volume_market_share_pct'),
        (pl.col('quote_volume') / pl.col('total_market_quote_volume') * pl.lit(100)).alias('quote_volume_market_share_pct'),
    ])

    df_with_totals = df_with_totals.drop(['total_market_volume', 'total_market_quote_volume'])

    return df_with_totals

def calculate_and_plot_factor_correlations(df: pl.DataFrame, factor_columns: list = None, figsize: tuple = (12, 10)):
    """
    计算指定因子列的相关性，并绘制热图。

    参数:
    df : pl.DataFrame
        包含因子数据的 Polars DataFrame
    factor_columns : list, optional
        要计算相关性的因子列名列表。如果为 None，则使用所有数值列。
    figsize : tuple, optional
        图形大小，默认为 (12, 10)

    返回:
    None (显示热图)
    """
    # 如果未指定因子列，则使用所有数值列
    if factor_columns is None:
        factor_columns = df.select(pl.all().exclude(['symbol', 'open_time', 'close_time'])).columns

    # 计算相关性矩阵
    corr_matrix = df.select(factor_columns).corr()

    # print (f'corr_matrix: {corr_matrix}')

    # print(f"相关性矩阵形状: {corr_matrix.shape}")
    # print(f"相关性矩阵列名: {corr_matrix.columns}")

    # 将相关性矩阵转换为可绘图的格式
    corr_data = []
    for i, row in enumerate(corr_matrix.rows()):
        for j, value in enumerate(row):
            if i != j:  # 排除对角线上的自相关
                corr_data.append((factor_columns[i], factor_columns[j], value))
            else:
                break

    corr_df = pl.DataFrame({
        'factor1': [x[0] for x in corr_data],
        'factor2': [x[1] for x in corr_data],
        'correlation': [x[2] for x in corr_data]
    })
    corr_df = corr_df.filter(pl.col('correlation').is_not_nan())
    corr_df = corr_df.sort(pl.col('correlation').abs(), descending=True)

    # print (f'corr_df: {corr_df.head (20)}')

    # 创建热图
    if 0:
        # 将 Polars DataFrame 转换为 Pandas DataFrame（Seaborn 需要）
        corr_matrix_pd = corr_matrix.to_pandas()
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix_pd, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Factor Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
    return corr_df


def select_low_correlation_factors(df: pl.DataFrame, factor_columns: list, correlation_threshold: float = 0.7, method: str = 'hierarchical'):
    """
    从一组因子中选择相关性较低的子集。

    参数:
    df : pl.DataFrame
        包含因子数据的 Polars DataFrame
    factor_columns : list
        要考虑的因子列名列表
    correlation_threshold : float, optional
        相关性阈值，默认为 0.7
    method : str, optional
        选择方法，可选 'hierarchical' 或 'greedy'，默认为 'hierarchical'

    返回:
    list : 选中的低相关性因子列表
    """
    # 计算相关性矩阵
    corr_matrix = df.select(factor_columns).corr().to_numpy()
    
    # 确保相关性矩阵是对称的
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    if method == 'hierarchical':
        # 使用层次聚类
        distance_matrix = 1 - np.abs(corr_matrix)
        # 确保距离矩阵是对称的
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        # 检查并处理可能的数值问题
        np.fill_diagonal(distance_matrix, 0)  # 确保对角线为0
        distance_matrix = np.clip(distance_matrix, 0, 2)  # 将值限制在[0, 2]范围内
        
        try:
            linkage = hierarchy.linkage(squareform(distance_matrix), method='complete')
            clusters = hierarchy.fcluster(linkage, t=1-correlation_threshold, criterion='distance')
        except ValueError as e:
            print(f"聚类过程中出错: {e}")
            print("转为使用贪婪算法")
            method = 'greedy'
        
        if method == 'hierarchical':
            # 从每个簇中选择一个代表因子
            selected_factors = []
            for cluster_id in np.unique(clusters):
                cluster_factors = [factor_columns[i] for i, c in enumerate(clusters) if c == cluster_id]
                # 选择簇中方差最大的因子
                variances = df.select(cluster_factors).var().to_numpy().flatten()
                selected_factors.append(cluster_factors[np.argmax(variances)])
    
    if method == 'greedy':
        # 贪婪算法
        selected_factors = [factor_columns[0]]  # 从第一个因子开始
        for factor in factor_columns[1:]:
            correlations = np.abs(corr_matrix[factor_columns.index(factor), [factor_columns.index(f) for f in selected_factors]])
            if np.all(correlations < correlation_threshold):
                selected_factors.append(factor)
    
    return selected_factors


if __name__ == "__main__":
    print(f"Polars version: {pl.__version__}")

    factor_file_path = "data/result_hour_alpha101.parquet"
    factor_data = pl.read_parquet(factor_file_path)
    
    # only derive single factor and drop auto_corr columns
    factor_data = factor_data.select(
        [col for col in factor_data.columns if "compound" not in col.lower() and "auto_corr" not in col.lower()]
    )
    print(factor_data)
   
    factor_data = factor_data.drop([ 'rolling_quote_volume_sum', ])

    factor_data = derive_basic_factors(factor_data)
    factor_data = derive_cross_sectional_factors(factor_data)
    print(factor_data.shape, factor_data.columns)
    print(factor_data)
    
    factor_data = factor_data.drop(
        [
            'volume',
            'quote_volume',
            'taker_buy_quote_volume',
            'taker_buy_volume',
            # 'count',
            'open',
            'high',
            'low',
            'close',
            'alpha49',
            'alpha33',
        ]
    )

    calculate_and_plot_factor_correlations(factor_data)
