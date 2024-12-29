import polars as pl
import os

dir_path = "hour_data/"

# List all files in the directory
files = os.listdir(dir_path)

# Initialize an empty list to hold DataFrames
dfs = []

# Loop through each file in the directory
for file in files:
    # Construct the full file path
    file_path = os.path.join(dir_path, file)
    cur_symbol = file.split("_")[0]

    # Read the file into a DataFrame
    df = pl.read_parquet(file_path)
    df = df.with_columns(pl.lit(cur_symbol).alias("symbol"))

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames into one
final_df = pl.concat(dfs)

columns_to_cast_float = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

col_cast_to_int = ["count", "ignore"]

final_df = final_df.with_columns(
    [pl.col(col).cast(pl.Float64) for col in columns_to_cast_float]
)

final_df = final_df.with_columns(
    [pl.col(col).cast(pl.Int64) for col in col_cast_to_int]
)

final_df = final_df.with_columns(
    pl.from_epoch(pl.col("open_time"), time_unit="ms")
    .cast(pl.Datetime("ms"))
    .alias("open_time"),
    pl.from_epoch(pl.col("close_time"), time_unit="ms")
    .cast(pl.Datetime("ms"))
    .alias("close_time"),
)


# Display the final DataFrame
print(final_df)

final_df.write_parquet("hour_data.parquet")
final_df.filter(pl.col("open_time") > pl.datetime(2024, 1, 1)).write_parquet(
    "small_hour_data.parquet"
)

print(f'small data: {pl.read_parquet("small_hour_data.parquet")}')
