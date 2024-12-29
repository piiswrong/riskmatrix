import polars as pl

def sign(column):
    return pl.when(column > 0).then(1).otherwise(pl.when(column < 0).then(-1).otherwise(0))