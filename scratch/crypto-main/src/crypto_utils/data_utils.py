import polars as pl

def convert_timestamp_col(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Convert a timestamp column from epoch milliseconds to datetime format.
    
    Args:
        df: Input DataFrame containing the timestamp column
        col_name: Name of the column containing epoch millisecond timestamps
        
    Returns:
        DataFrame with the timestamp column converted to datetime
    """
    return df.with_columns(pl.from_epoch(pl.col(col_name), time_unit="ms").cast(pl.Datetime("ms")).alias(col_name))