import pandas as pd
import numpy as np

def verify_incremental_data(file_path):
    """验证增量数据的各项指标"""
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    
    # 1. 基本数据完整性检查
    def check_data_integrity():
        # 检查时间连续性
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
        df['time_diff'] = df['timestamp'].diff() / 1000  # 转换为毫秒
        
        time_gaps = df[df['time_diff'] > 1000].copy()  # 超过1秒的间隔
        
        return {
            'total_records': len(df),
            'time_range': {
                'start': df['datetime'].min(),
                'end': df['datetime'].max(),
                'duration': str(df['datetime'].max() - df['datetime'].min())
            },
            'time_gaps': len(time_gaps),
            'max_gap_ms': df['time_diff'].max(),
            'avg_gap_ms': df['time_diff'].mean()
        }
    
    # 2. 价格合理性检查
    def check_price_validity():
        # 分别检查bid和ask的价格变化
        price_stats = df.groupby('side').agg({
            'price': ['min', 'max', 'std', 'count']
        })
        
        # 检查价格跳变
        df['price_change'] = df.groupby('side')['price'].pct_change()
        large_changes = df[abs(df['price_change']) > 0.01].copy()  # 1%以上的价格变化
        
        return {
            'price_stats': price_stats,
            'large_changes': len(large_changes),
            'max_change_pct': df['price_change'].max() * 100
        }
    
    # 3. 订单簿完整性检查
    def check_orderbook_integrity():
        # 检查买卖双方的数量平衡
        side_balance = df['side'].value_counts()
        
        # 检查amount为0的更新（删除操作）
        deletes = df[df['amount'] == 0]
        
        return {
            'side_balance': side_balance,
            'delete_operations': len(deletes),
            'delete_percentage': len(deletes) / len(df) * 100
        }
    
    # 4. 更新频率分析
    def check_update_frequency():
        df['second'] = df['datetime'].dt.floor('S')
        updates_per_second = df.groupby('second').size()
        
        return {
            'updates_per_second': {
                'mean': updates_per_second.mean(),
                'median': updates_per_second.median(),
                'max': updates_per_second.max(),
                'min': updates_per_second.min()
            }
        }
    
    # 执行所有检查
    results = {
        'data_integrity': check_data_integrity(),
        'price_validity': check_price_validity(),
        'orderbook_integrity': check_orderbook_integrity(),
        'update_frequency': check_update_frequency()
    }
    
    # 打印结果
    print("\n=== Data Integrity ===")
    print(f"Total Records: {results['data_integrity']['total_records']}")
    print(f"Time Range: {results['data_integrity']['time_range']['start']} to {results['data_integrity']['time_range']['end']}")
    print(f"Duration: {results['data_integrity']['time_range']['duration']}")
    print(f"Time Gaps > 1s: {results['data_integrity']['time_gaps']}")
    
    print("\n=== Price Statistics ===")
    print(results['price_validity']['price_stats'])
    print(f"Large Price Changes (>1%): {results['price_validity']['large_changes']}")
    
    print("\n=== Order Book Balance ===")
    print(results['orderbook_integrity']['side_balance'])
    print(f"Delete Operations: {results['orderbook_integrity']['delete_operations']}")
    
    print("\n=== Update Frequency ===")
    freq = results['update_frequency']['updates_per_second']
    print(f"Updates per Second: Mean={freq['mean']:.2f}, Median={freq['median']:.2f}, Max={freq['max']}")
    
    return results

if __name__ == "__main__":
    file_path = '/home/dp/backup_crypto/crypto/data/HFT/202401/binance-futures_incremental_book_L2_2024-01-10_RIFUSDT.parquet'
    results = verify_incremental_data(file_path)