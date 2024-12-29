"""
This is a position manager class.
"""

import os
import polars as pl
from typing import List

class Position:
    # open only
    def __init__(self, symbol: str, transaction_time: int = 0, price: float = 0, side: str = 'BUY', quantity: float = 0):
        self.symbol = symbol
        self.transaction_time = transaction_time
        self.price = price
        self.side = side
        self.quantity = quantity

    def to_dict(self):
        return self.__dict__

    def get_net_quantity(self):
        return self.quantity if self.side == 'BUY' else -self.quantity

    def __add__(self, other):
        if isinstance(other, Position) and self.symbol == other.symbol:
            new_quantity = self.get_net_quantity() + other.get_net_quantity()
            if new_quantity == 0:
                return Position(self.symbol)

            new_side = self.side if new_quantity >= 0 else ('SELL' if self.side == 'BUY' else 'BUY')
            new_price = (self.price * self.get_net_quantity() + other.price * other.get_net_quantity()) / new_quantity
            return Position(self.symbol, max(self.transaction_time, other.transaction_time), new_price, new_side, abs(new_quantity))
        else:
            raise TypeError("Can only add Position objects with the same symbol")

    def __sub__(self, other):
        if isinstance(other, Position) and self.symbol == other.symbol:
            return self + Position(other.symbol, other.transaction_time, other.price, 'SELL' if other.side == 'BUY' else 'BUY', other.quantity)
        else:
            raise TypeError("Can only subtract Position objects with the same symbol")

class PositionManager:
    def __init__(self, file: str):
        self.file = file
        self.active_positions = {}
        self.__load_positions()

    def __load_positions(self):
        if os.path.exists(self.file):
            df = pl.read_csv(self.file, raise_if_empty=False)
            positions = [Position(**position) for position in df.to_dicts()]
            self.__parse_positions(positions)

            print(f"Loaded {len(self.get_open_positions())} positions from {self.file}")
        else:
            print(f"No positions found in {self.file}")

    def __parse_positions(self, positions: List[Position]):
        for position in positions:
            self.add_position(position)

        # delete empty positions
        self.active_positions = {k: v for k, v in self.active_positions.items() if v.quantity != 0}


    def add_position(self, position: Position):
        if position is None:
            return

        self.active_positions[position.symbol] = self.active_positions.get(position.symbol, Position(position.symbol)) + position

        if self.active_positions[position.symbol].quantity == 0:
            del self.active_positions[position.symbol]


    def get_open_positions(self):
        return list(self.active_positions.values())


    def save(self):
        df = pl.DataFrame([position.to_dict() for position in self.get_open_positions()])
        df.write_csv(self.file)
        print(f"Saved {len(self.get_open_positions())} positions to {self.file}")
