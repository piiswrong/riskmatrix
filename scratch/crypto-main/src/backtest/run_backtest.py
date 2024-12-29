from backtest.backtest import main_backtest
from backtest.config import BacktestConfig
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='运行回测策略')

    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='预测数据输入路径，例如: gubo/data/alpha1_pred.parquet')

    parser.add_argument('--day_num',
                        type=int,
                        required=True,
                        help='调仓的周期')

    parser.add_argument('--factor_prefix',
                        type=str,
                        default='linear_compound_factor',
                        help='因子名称前缀，默认为 linear_compound_factor')

    return parser.parse_args()


def main():
    args = parse_args()

    config = BacktestConfig(
        input_df_path=args.input_path,
        delist_info_path='backtest/delist/delist_info.csv',
        update_position_time=args.day_num,
        trade_factor_name=f"{args.factor_prefix}_{args.day_num}day",
        start_date=datetime(2024, 6, 1),
    )

    main_backtest(config)


if __name__ == "__main__":
    main()
