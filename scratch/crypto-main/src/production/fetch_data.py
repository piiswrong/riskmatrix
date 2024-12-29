from binance.client import Client
import sys
import datetime
import polars as pl
from tqdm import tqdm
import concurrent.futures

def get_all_futures_symbols():
    client = Client("", "")
    return client.futures_exchange_info()['symbols']

def get_klines_of_date(symbol: str, epoch: int):
    client = Client("", "")
    return client.futures_klines(symbol=symbol, interval="1d", startTime=epoch, limit=1)

def parse_klines(kline_response: list, symbol: str):
    """
        [
        1591258320000,          // Open time
        "9640.7",               // Open
        "9642.4",               // High
        "9640.6",               // Low
        "9642.0",               // Close (or latest price)
        "206",                  // Volume
        1591258379999,          // Close time
        "2.13660389",           // Base asset volume
        48,                     // Number of trades
        "119",                  // Taker buy volume
        "1.23424865",           // Taker buy base asset volume
        "0"                     // Ignore.
        ]
    """
    d = {
        "symbol": symbol,
        "open_time": kline_response[0],
        "open": float(kline_response[1]),
        "high": float(kline_response[2]),
        "low": float(kline_response[3]),
        "close": float(kline_response[4]),
        "volume": float(kline_response[5]),
        "close_time": kline_response[6],
        "quote_volume": float(kline_response[7]),
        "count": int(kline_response[8]),
        "taker_buy_volume": float(kline_response[9]),
        "taker_buy_quote_volume": float(kline_response[10]),
    }
    return d

def process_symbol(symbol, epoch):
    try:
        klines = get_klines_of_date(symbol['symbol'], epoch)
        if len(klines) == 0:
            print(f"No klines found for {symbol['symbol']}")
            return None
        if klines[0][0] != epoch:
            print(klines[0])
            print(epoch)
            print(f"Kline open time is not {epoch} for {symbol['symbol']}")
            return None
        return parse_klines(klines[0], symbol['symbol'])
    except Exception as e:
        print(f"Error fetching data for {symbol['symbol']}: {e}")
        return None

def append_result(path, new_result):
    old_df = pl.read_parquet(path).select(["symbol", "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"])
    new_result = pl.DataFrame(new_result).select(["symbol", "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"])
    new_df = pl.concat([old_df, new_result]).unique(subset=["symbol", "open_time"]).filter(pl.col("volume") > 0)
    new_df.write_parquet(path)

def main(date_str, num_threads, output_path):
    date = datetime.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=datetime.timezone.utc)
    epoch = int(date.timestamp() * 1000)
    print("epoch: ", epoch)
    
    symbols = get_all_futures_symbols()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_symbol = {executor.submit(process_symbol, symbol, epoch): symbol for symbol in symbols}
        for future in tqdm(concurrent.futures.as_completed(future_to_symbol), total=len(symbols)):
            result = future.result()
            if result:
                results.append(result)
    
    df = pl.DataFrame(results).unique(["symbol", "open_time"])
    append_result(output_path, df)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python fetch_data.py <date> <num_threads> <output_path>")
        sys.exit(1)

    date_str = sys.argv[1]
    if date_str == "today":
        date_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    num_threads = int(sys.argv[2])
    output_path = sys.argv[3]
    main(date_str, num_threads, output_path)