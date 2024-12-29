import pandas as pd
import numpy as np

def validate_and_process_trades(df):
    """
    验证交易数据并生成1分钟K线
    """
    # 1. 基础数据检查
    print("数据基本检查:")
    print(f"总行数: {len(df)}")
    print(f"时间范围: {pd.to_datetime(df['timestamp'].min(), unit='us')} 至 {pd.to_datetime(df['timestamp'].max(), unit='us')}")
    print(f"交易对: {df['symbol'].unique()}")
    print(f"价格范围: {df['price'].min()} - {df['price'].max()}")
    print("\n异常值检查:")
    
    # 2. 检查空值
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("警告：存在空值")
        print(null_counts[null_counts > 0])
    else:
        print("✓ 无空值")
    
    # 3. 检查价格和数量是否为正
    invalid_price = df[df['price'] <= 0].shape[0]
    invalid_amount = df[df['amount'] <= 0].shape[0]
    print(f"✓ 负价格数量: {invalid_price}")
    print(f"✓ 负数量成交: {invalid_amount}")
    
    # 4. 检查时间戳排序
    is_sorted = (df['timestamp'].diff() >= 0).all()
    print(f"✓ 时间戳单调递增: {is_sorted}")
    
    # 5. 生成1分钟K线
    # 将时间戳转换为datetime并向下取整到分钟
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
    df['minute'] = df['datetime'].dt.floor('T')
    
    # 按分钟聚合
    klines = df.groupby('minute').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    }).reset_index()
    
    # 规范列名
    klines.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    return klines

if __name__ == "__main__":
    file_path = '/home/dp/backup_crypto/crypto/data/HFT/202401/binance-futures_trades_2024-01-04_JOEUSDT.parquet'
    df = pd.read_parquet(file_path)
    print (df)

    klines_df = validate_and_process_trades(df)

    # 打印K线数据样例
    print("\n生成的1分钟K线示例:")
    print(klines_df.head())