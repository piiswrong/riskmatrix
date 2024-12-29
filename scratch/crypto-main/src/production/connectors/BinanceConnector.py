import yaml
from binance.client import Client
from binance.enums import ORDER_TYPE_MARKET, SIDE_BUY, SIDE_SELL
from position import Position
from binance.helpers import round_step_size

class BinanceFuturesConnector:
    def __init__(self, config):
        self.config = config['binance_connector']

        self.api_key = self.config['api_key']
        self.secret_key = self.config['secret_key']
        self.is_testnet = self.config['is_testnet']
        self.client = Client(self.api_key, self.secret_key, testnet=self.is_testnet)


    def get_last_price(self, symbol: str):
        return float(self.client.futures_symbol_ticker(symbol=symbol)['price'])

    def get_tradable_size(self, symbol: str, usdt_qty: float):
        last_price = self.get_last_price(symbol)
        return self._adjust_quantity_precision(symbol, usdt_qty / last_price)

    def get_step_size(self, symbol: str):
        exchange_info = self.client.futures_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
        if symbol_info:
            step_size = symbol_info['filters'][1]['stepSize']
            return step_size
        else:
            raise Exception(f"Symbol {symbol} not found")

    def _adjust_quantity_precision(self, symbol: str, quantity: float):
        step_size = self.get_step_size(symbol)
        return round_step_size(quantity, step_size)

    def fetch_total_margin(self):
        try:
            margin = self.client.futures_account()['totalMarginBalance']
            return float(margin)
        except Exception as e:
            raise e

    def open_position(self, symbol: str, side: str, quantity: float, leverage: int):
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity * leverage
            )
            print(f"Executed order: {order}")
            position = Position(symbol, int(order['updateTime']), float(order['price']), side, quantity)
            return position
        except Exception as e:
            print(f"An error occurred: {e}, for symbol: {symbol}, side: {side}, quantity: {quantity}")
            return None

    def close_position(self, symbol: str, side: str, quantity: float):
        try:
            # To close a position, we need to place an order in the opposite direction
            close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
            order = self.client.futures_create_order(
                symbol=symbol,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            position = Position(symbol, int(order['updateTime']), float(order['price']), close_side, quantity)
            print(f"Executed order: {order}")
            return position
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
