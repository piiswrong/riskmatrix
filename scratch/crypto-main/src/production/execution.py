import yaml
import polars as pl
from position import PositionManager
from connectors.IOConnector import IOConnector
from connectors.BinanceConnector import BinanceFuturesConnector
from enum import Enum
import sys
import os
from config import Config
from datetime import datetime, timedelta

class Operation(Enum):
    LONG = 1
    SHORT = 2

class Execution:
    def __init__(self, config):
        self.config = config["execution"]

        print("configs: ", self.config)

        self.date = self.config['date']

        if self.date == "today":
            self.date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            print("Using yesterday's date: ", self.date)
        elif self.date == "DATE":
            self.date = os.getenv("DATE")
            print("Using date from environment variable: ", self.date)

        self.remainder = (int(self.date) % 5)

        position_file = self.config['position_file'] + f"_{self.remainder}.csv"
        print("Using position file: ", position_file)

        self.position_manager = PositionManager(position_file)
        self.prediction_file = self.config['prediction_file']
        self.prediction_file = self.prediction_file.replace("DATE", self.date)

        self.long_condition = eval(self.config['long_condition'])
        self.short_condition = eval(self.config['short_condition'])
        self.rank_col = self.config['rank_col']
        self.single_order_qty_unit_long = self.config['single_order_qty_unit_long']
        self.single_order_qty_unit_short = self.config['single_order_qty_unit_short']
        self.qty_unit_ratio = self.config['qty_unit_ratio']
        self.margin_usage_ratio = self.config['margin_usage_ratio']
        self.long_scale_factor = self.config['long_scale_factor']
        self.short_scale_factor = self.config['short_scale_factor']
        self.vol_quantile = self.config['vol_quantile']
        self.vol_col = self.config['vol_col']
        self.leverage = self.config['leverage']

        exclude_symbols_file = self.config['exclude_symbols']
        self.exclude_symbols = pl.read_csv(exclude_symbols_file).select(pl.col("symbol")).to_numpy().flatten()

        print("exclude_symbols: ", self.exclude_symbols)

        self.connector = None

    def set_connector(self, connector):
        self.connector = connector

    def fetch_qty_unit(self):
        total_margin = self.connector.fetch_total_margin()
        unit_in_usdt = total_margin * self.margin_usage_ratio * self.qty_unit_ratio
        self.single_order_qty_in_usdt_long = self.single_order_qty_unit_long * unit_in_usdt
        self.single_order_qty_in_usdt_short = self.single_order_qty_unit_short * unit_in_usdt

        print(f"total margin: {total_margin}, open unit_in_usdt: {unit_in_usdt}")
        print(f"open single_order_qty_in_usdt_long: {self.single_order_qty_in_usdt_long}")
        print(f"open single_order_qty_in_usdt_short: {self.single_order_qty_in_usdt_short}")

    def __load_prediction(self):
        self.prediction = (
            pl.read_parquet(self.prediction_file)
              .sort("open_time")
              .filter(pl.col(self.rank_col).is_not_null())
              .filter(pl.col("symbol").str.ends_with("USDT"))
              .filter(~pl.col("symbol").is_in(self.exclude_symbols))
        )
        self.prediction = self.prediction.with_columns(
            pl.from_epoch(pl.col('open_time'), time_unit='ms').dt.strftime("%Y%m%d").alias('date')
        )

        self.prediction = self.prediction.filter(
            pl.col('date') == self.date
        )

        quantile_filter = self.prediction.select(pl.col(self.vol_col).quantile(self.vol_quantile))
        self.prediction = self.prediction.filter(pl.col(self.vol_col) > quantile_filter)

        # Convert the date string to datetime and filter
        self.prediction = self.prediction.with_columns(
            [
                pl.col(self.rank_col).rank().over('open_time').alias('reverse_rank'),
                pl.col(self.rank_col).rank(descending=True).over('open_time').alias('rank'),
            ]
        ).sort(['open_time', 'reverse_rank'])


        self.__check_date()


    def __get_tradable_symbols(self, operation):
        if operation == Operation.LONG:
            return self.prediction.filter(self.long_condition).select("symbol").to_numpy().flatten()
        elif operation == Operation.SHORT:
            return self.prediction.filter(self.short_condition).select("symbol").to_numpy().flatten()

    def __get_scale_factor(self, operation: Operation, symbol: str):
        return 1

    def __check_date(self):
        if self.prediction.is_empty():
            raise Exception(f"No prediction for {self.date}, skip execution")
        if self.prediction['date'].max() < self.date:
            raise Exception(f"No prediction for {self.date}, skip execution")

    def __close_all_positions(self):
        for position in self.position_manager.get_open_positions():
            print(f"Closing position for {position.symbol}, quantity: {position.quantity}")
            position = self.connector.close_position(position.symbol, position.side, position.quantity)
            if position is not None:
                self.position_manager.add_position(position)

    def __execute_predictions(self):
        long_symbols = self.__get_tradable_symbols(Operation.LONG)
        short_symbols = self.__get_tradable_symbols(Operation.SHORT)

        print(f"Long symbols: {long_symbols}")
        print(f"Short symbols: {short_symbols}")

        for symbol in long_symbols:
            last_price = self.connector.get_last_price(symbol)
            scale_factor = self.__get_scale_factor(Operation.LONG, symbol)
            tradable_size = self.connector.get_tradable_size(symbol, self.single_order_qty_in_usdt_long * scale_factor)
            print(f"Opening long position for {symbol}, last price: {last_price}, quantity: {tradable_size}, scale factor: {scale_factor}")
            position = self.connector.open_position(symbol, "BUY", tradable_size, self.leverage)

            if position is not None:
                self.position_manager.add_position(position)
            else:
                print(f"Failed to open long position for {symbol}")

        for symbol in short_symbols:
            last_price = self.connector.get_last_price(symbol)
            scale_factor = self.__get_scale_factor(Operation.SHORT, symbol)
            tradable_size = self.connector.get_tradable_size(symbol, self.single_order_qty_in_usdt_short * scale_factor)
            print(f"Opening short position for {symbol}, last price: {last_price}, quantity: {tradable_size}, scale factor: {scale_factor}")
            position = self.connector.open_position(symbol, "SELL", tradable_size, self.leverage)

            if position is not None:
                self.position_manager.add_position(position)
            else:
                print(f"Failed to open short position for {symbol}")

        self.position_manager.save()

    
    def run(self):
        self.__load_prediction()
        self.__close_all_positions()
        self.__execute_predictions()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execution.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config = Config(config_file).config
    execution = Execution(config)
    io_connector = IOConnector(config)
    binance_connector = BinanceFuturesConnector(config)

    # execution.set_connector(binance_connector)
    execution.set_connector(io_connector)
    execution.fetch_qty_unit()

    execution.run()
