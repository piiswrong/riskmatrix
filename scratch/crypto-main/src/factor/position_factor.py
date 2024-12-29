import numpy as np
import polars as pl
import pandas as pd
import talib
from datetime import datetime
import pandas_ta


def factor1(close, window=250):
    # 251, window +1
    # window=285 #[258, 0.625，0.82], [280，0.656，0.82] 285
    f = (
        talib.STDDEV(close, window)
        / talib.ATR(close, close, close, window)
        / np.sqrt(window)
    )
    signal = 1.0 - f
    signal[signal < 0.67] = 0.0
    signal[signal > 0.82] = 0.82
    return signal / 0.82


def factor2(close, window1=32, window2=200):
    # window, 200
    f = talib.STDDEV(close, window1) / talib.STDDEV(close, window2)
    f[f < 0.55] = 0.0
    f[f > 0.85] = 0.85
    return f / 0.85


def factor3(close, window=240):
    f = (
        (talib.MAX(close, window) - talib.MIN(close, window))
        / talib.ATR(close, close, close, window)
    ) / np.sqrt(window)
    f = f / 5
    f = 1 - f
    f[f < 0.6] = 0
    f[f > 0.8] = 0.8
    return f / 0.8


def factor4(close, window=300):
    # window, 300
    difference = abs(np.diff(close, 2))
    difference = np.r_[np.zeros(len(close) - len(difference)), difference]
    difference = pd.Series(difference, index=close.index).fillna(0)
    f = talib.SUM(difference, window) / (
        talib.MAX(close, window) - talib.MIN(close, window)
    )
    signal = f / np.sqrt(window) / 2.5
    signal[signal < 0.3] = 0.0
    signal[signal > 0.55] = 0.55
    return signal / 0.55


def factor5(
    close,
    window1=30,
    window2=300,
    window3=35,
    window4=300,
    window5=30,
    window6=280,
    window7=35,
    window8=340,
    window9=30,
    window10=340,
):
    # 2*window, 640
    # window1 = 30
    # window2 = 300
    minus = (talib.EMA(close, window1) - talib.EMA(close, window2)) / talib.ATR(
        close, close, close, window2
    )
    f1 = 1.0 - talib.STDDEV(minus, window2) / np.sqrt(window2)

    # window3 = 35
    # window4 = 300
    minus = (talib.EMA(close, window3) - talib.EMA(close, window4)) / talib.ATR(
        close, close, close, window4
    )
    f2 = 1.0 - talib.STDDEV(minus, window4) / np.sqrt(window4)

    # window5 = 30
    # window6 = 280
    minus = (talib.EMA(close, window5) - talib.EMA(close, window6)) / talib.ATR(
        close, close, close, window6
    )
    f3 = 1.0 - talib.STDDEV(minus, window6) / np.sqrt(window6)

    # window7 = 35
    # window8 = 340
    minus = (talib.EMA(close, window7) - talib.EMA(close, window8)) / talib.ATR(
        close, close, close, window8
    )
    f4 = 1.0 - talib.STDDEV(minus, window8) / np.sqrt(window8)

    # window9 = 30
    # window10 = 340
    minus = (talib.EMA(close, window9) - talib.EMA(close, window10)) / talib.ATR(
        close, close, close, window10
    )
    f5 = 1.0 - talib.STDDEV(minus, window10) / np.sqrt(window10)

    f = (f1 + f2 + f3 + f4 + f5) / 5
    f[f < 0.8] = 0.0
    f[f > 1] = 1
    return f


def factor6(close, window1=192, window2=1200):
    # 1391
    # window1 = 96*2
    # window2 = 1200  # 1200
    std = talib.STDDEV(close, window1)
    ave = talib.SMA(std, window2)
    stdd = talib.STDDEV(std, window2)
    stdd[stdd == 0] = np.nan
    f = 3 - (std - ave) / stdd
    f[f < 3.9] = 0
    f[f > 5] = 5
    f = f / 5
    return f


def factor7(close, window=250):
    # 250
    cl = np.log(close)
    yy = np.append(np.array([0]), np.diff(cl))  # cl-ma
    yy[np.isnan(yy)] = 0
    zz = yy.cumsum()

    # window = 250
    yst = talib.STDDEV(yy, window)
    zst = talib.STDDEV(zz, window)
    yst[yst == 0] = 1000
    hurst1 = zst / yst
    f = 11 - hurst1
    f[f < 7.1] = 0
    f /= 11
    return pd.Series(f, index=close.index)


def factor8(close, window=220):
    # 234
    rsi = talib.RSI(close, 14 * 1)
    f = np.abs(talib.CORREL(close, rsi, window))  # int(15*6.5)
    f[f < 0.65] = 0
    f[f > 0.9] = 0.9
    f /= 0.9
    return f


def factor9(close, window1=75, window2=300, window3=300):
    # 608
    r1 = talib.RSI(close, int(window1))
    r2 = talib.RSI(close, int(window2))
    f = talib.SMA(np.abs(r1 - r2), window3)
    f = 10 - f
    f[f < 8] = 0
    f = f / 10
    return f


def get_signals(df):
    # Calculate indicators and trading signals
    factors = 0
    close = df["close"]

    # Factor 1
    factors += factor1(close, window=20)

    # Factor 2
    factors += factor2(close, window1=5, window2=20)

    # Factor 3
    factors += factor3(close, window=24)

    # print (f'input: {df.shape}, output: {factors.shape} {type (factors)}')
    # print (f'input:\n{df} \n output: {factors}')
    # return factors

    # Factor 4
    factors += factor4(close, window=30)

    # Factor 5
    factors += factor5(
        close,
        window1=5,
        window2=30,
        window3=14,
        window4=30,
        window5=5,
        window6=30,
        window7=5,
        window8=30,
        window9=5,
        window10=30,
    )

    # Factor 6
    # factors += factor6(close)

    # Factor 7
    factors += factor7(close, window=30)

    # Factor 8
    factors += factor8(close, window=30)

    # Factor 9
    factors += factor9(close, window1=5, window2=30, window3=30)

    vol = talib.STDDEV(close / close.shift() - 1, 20)

    return factors/vol


def CalcPositionFactor(all_time_hist_data: pl.DataFrame) -> pl.DataFrame:
    # Convert to pandas DataFrame
    df = all_time_hist_data.to_pandas()
    df.set_index(["symbol", "open_time"], inplace=True)
    df.sort_index(inplace=True)

    # Group by symbol, sort by open_time, and apply get_signals
    results = df.groupby(level="symbol").apply(
        lambda x: get_signals(x.reset_index(level=["symbol"], drop=True))
    )

    # print('all time shape: ', all_time_hist_data.shape)
    # print(type(results), results.shape)

    # If get_signals returns a Series, convert it to a DataFrame
    if isinstance(results, pd.Series):
        results = results.to_frame(name="pos_signals")

    # Reset index of results to match the original DataFrame's index
    results = results.reset_index()

    # Merge the results back with the original data
    df_reset = df.reset_index()
    df_with_signals = pd.merge(
        df_reset, results, on=["symbol", "open_time"], how="left"
    )

    # Set back the index if needed
    df_with_signals.set_index(["symbol", "open_time"], inplace=True)

    # Convert df_with_signals back to a Polars DataFrame
    result = pl.from_pandas(df_with_signals.reset_index())

    return result


if __name__ == "__main__":
    history_data_path = "data/all_data_1d_boris_converted.parquet"
    all_time_hist_data = pl.read_parquet(history_data_path)
    result = CalcPositionFactor(all_time_hist_data)
    print(result)
