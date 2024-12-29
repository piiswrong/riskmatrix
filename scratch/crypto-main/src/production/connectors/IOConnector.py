import yaml
import json
from datetime import datetime
from position import Position


class IOConnector:
    def __init__(self, config):
        self.config = config["io_connector"]
        self.log_file = self.config["io_connector_file"]

    def _log_action(self, action: str, details: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        }

        try:
            with open(self.log_file, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            print(f"Error logging action: {e}")

    def get_last_price(self, symbol: str):
        return 2

    def get_tradable_size(self, symbol: str, usdt_qty: float):
        return usdt_qty / self.get_last_price(symbol)

    def open_position(self, symbol: str, side: str, quantity: float, leverage: int):
        details = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "type": "MARKET",
            "timestamp": datetime.now().timestamp(),
            "price": 0,
            "leverage": leverage,
        }
        self._log_action("open_position", details)
        position = Position(
            symbol, details["timestamp"], details["price"], side, quantity
        )
        return position

    def close_position(self, symbol: str, side: str, quantity: float):
        close_side = "SELL" if side == "BUY" else "BUY"
        details = {
            "symbol": symbol,
            "side": close_side,
            "quantity": quantity,
            "type": "MARKET",
            "timestamp": int(datetime.now().timestamp()),
            "price": 0,
        }
        self._log_action("close_position", details)
        position = Position(
            symbol, details["timestamp"], details["price"], close_side, quantity
        )
        return position

    def fetch_total_margin(self):
        return 3000