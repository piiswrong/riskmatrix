import pandas as pd
import numpy as np
from typing import Tuple, Set

# Global configuration
PRECISION = 6


def round_to_precision(series: pd.Series) -> pd.Series:
    """
    Round numeric values to specified precision.
    Leaves non-numeric values unchanged.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.round(PRECISION)
    return series

    
def compare_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> Set[str]:
    """
    Compare column sets between DataFrames.
    Returns the set of common columns.
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    if cols1 != cols2:
        print("\nColumn differences found:")
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1

        if only_in_1:
            print(f"Columns only in File 1: {sorted(only_in_1)}")
        if only_in_2:
            print(f"Columns only in File 2: {sorted(only_in_2)}")

    return cols1.intersection(cols2)


def compare_column_types(
    df1: pd.DataFrame, df2: pd.DataFrame, common_columns: Set[str]
) -> None:
    """
    Compare data types for each common column.
    """
    print("\nComparing column types:")
    for column in sorted(common_columns):
        type1 = df1[column].dtype
        type2 = df2[column].dtype
        if type1 != type2:
            print(f"Column '{column}' has different types:")
            print(f"  File 1: {type1}")
            print(f"  File 2: {type2}")
        else:
            print(f"Column '{column}': {type1} (same TYPE in both files)")


def compare_line_numbers(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare DataFrames based on open_time column and filter to matching time interval.
    Shows rows unique to each file at mismatched timestamps.
    
    Args:
        df1 (pd.DataFrame): First DataFrame with open_time column
        df2 (pd.DataFrame): Second DataFrame with open_time column
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Filtered DataFrames containing only data 
        within the overlapping time interval
    """
    # First check if both DataFrames have open_time column
    if 'open_time' not in df1.columns or 'open_time' not in df2.columns:
        raise ValueError("Both DataFrames must have 'open_time' column")
        
    # Ensure open_time is datetime type
    df1['open_time'] = pd.to_datetime(df1['open_time'], unit='ms')
    df2['open_time'] = pd.to_datetime(df2['open_time'], unit='ms')
    
    # Get time ranges
    start_time1, end_time1 = df1['open_time'].min(), df1['open_time'].max()
    start_time2, end_time2 = df2['open_time'].min(), df2['open_time'].max()
    
    print(f"\nTime ranges:")
    print(f"File 1: {start_time1} to {end_time1}")
    print(f"File 2: {start_time2} to {end_time2}")
    
    # Find overlapping interval
    start_time = max(start_time1, start_time2)
    end_time = min(end_time1, end_time2)
    
    # temp debug
    # print (type (end_time), end_time)
    # DEBUG_END_TIME = pd.Timestamp ('2024-11-10 00:00:00')
    # end_time = DEBUG_END_TIME
    # print (f'set end_time to {DEBUG_END_TIME} for temp debug')

    # Filter both DataFrames to the overlapping interval
    df1_filtered = df1[(df1['open_time'] >= start_time) & (df1['open_time'] <= end_time)]
    df2_filtered = df2[(df2['open_time'] >= start_time) & (df2['open_time'] <= end_time)]
    
    print(f"\nFiltering both files to overlapping interval:")
    print(f"Interval: {start_time} to {end_time}")
    print(f"\nAfter filtering:")
    print(f"File 1: {len(df1_filtered)} rows")
    print(f"File 2: {len(df2_filtered)} rows")
    
    # If row counts differ within the same time range, analyze the differences
    if len(df1_filtered) != len(df2_filtered):
        print("\nWARNING: Different number of rows within the same time interval!")
        
        # Group by open_time and count rows in each DataFrame
        count1 = df1_filtered.groupby('open_time').size()
        count2 = df2_filtered.groupby('open_time').size()
        
        # Find timestamps where counts differ
        all_times = pd.concat([count1, count2], axis=1).fillna(0)
        all_times.columns = ['count1', 'count2']
        diff_times = all_times[all_times['count1'] != all_times['count2']]
        
        print("\nTimestamps with different row counts:")
        for time, row in diff_times.iterrows():
            print(f"\nTimestamp: {time}")
            print(f"File 1 rows: {int(row['count1'])}")
            print(f"File 2 rows: {int(row['count2'])}")
            
            # Get rows for this timestamp from both files
            df1_time = df1_filtered[df1_filtered['open_time'] == time]
            df2_time = df2_filtered[df2_filtered['open_time'] == time]
            
            # Find unique symbols in each file
            symbols1 = set(df1_time['symbol'])
            symbols2 = set(df2_time['symbol'])
            
            # Show symbols unique to file 1
            unique_to_file1 = symbols1 - symbols2
            if unique_to_file1:
                print("\nSymbols only in File 1:")
                unique_rows1 = df1_time[df1_time['symbol'].isin(unique_to_file1)]
                print(unique_rows1[['symbol']].to_string())
            
            # Show symbols unique to file 2
            unique_to_file2 = symbols2 - symbols1
            if unique_to_file2:
                print("\nSymbols only in File 2:")
                unique_rows2 = df2_time[df2_time['symbol'].isin(unique_to_file2)]
                print(unique_rows2[['symbol']].to_string())
    
    return df1_filtered, df2_filtered

def compare_column_values(df1: pd.DataFrame, df2: pd.DataFrame, column: str) -> None:
    """
    Compare values in a specific column between DataFrames, properly handling NaN values
    and different indices. Shows open_time and symbol for each mismatch.
    """
    try:
        # Convert series to numpy arrays to bypass index alignment issues
        array1 = np.array(round_to_precision(df1[column]))
        array2 = np.array(round_to_precision(df2[column]))
        
        # Handle NaN values using numpy's isnan
        na1 = np.isnan(array1) if np.issubdtype(array1.dtype, np.number) else np.array([pd.isna(x) for x in array1])
        na2 = np.isnan(array2) if np.issubdtype(array2.dtype, np.number) else np.array([pd.isna(x) for x in array2])
        
        # Values are different if:
        # 1. One is NaN and other isn't, or
        # 2. Neither are NaN and values are different
        mismatches = (na1 != na2) | (~na1 & ~na2 & (array1 != array2))
        mismatch_count = np.sum(mismatches)
        total_rows = len(df1)
        
        if mismatch_count == 0:
            print(f"\nColumn '{column}': 0 mismatches in {total_rows} total rows")
        else:
            print(
                f"\nColumn '{column}': {mismatch_count} mismatches in {total_rows} total rows"
            )
            print("First 5 mismatches:")
            # Get indices where values differ
            mismatch_indices = np.where(mismatches)[0][:5]
            for idx in mismatch_indices:
                print(f"Row {idx}:")
                val1 = array1[idx]
                val2 = array2[idx]
                open_time = df1['open_time'].iloc[idx]
                symbol = df1['symbol'].iloc[idx]
                print(f"  Symbol: {symbol}")
                print(f"  Open Time: {open_time}")
                if np.issubdtype(array1.dtype, np.number):
                    print(f"  File 1: {val1:.{PRECISION}f}")
                    print(f"  File 2: {val2:.{PRECISION}f}")
                else:
                    print(f"  File 1: {val1}")
                    print(f"  File 2: {val2}")
    except Exception as e:
        print(f"Error comparing column '{column}': {str(e)}")

def main():
    # Set file paths
    # file1_path = "/home/dp/backup_crypto/crypto/production/predictions.parquet"
    # file1_path = "/home/dp/backup_crypto/crypto/data/predictions_20241201.parquet"
    # file1_path = "/home/dp/backup_crypto/crypto/data/improve_predictions_20241201.parquet"
    file1_path = "/home/dp/backup_crypto/crypto/data/predictions_20241129.parquet"
    # file2_path = "/home/dp/backup_crypto/crypto/production/z790_predictions.parquet"
    # file2_path = "/home/dp/backup_crypto/crypto/production/old_pred.parquet"
    file2_path = "/home/dp/backup_crypto/crypto/production/z790_pred.parquet"

    exclude_symbols = ['BTCUSDT', 'ETHUSDT', 'BCHUSDT', 'LTCUSDT', 'ETCUSDT', 'LINKUSDT', 'AVAXUSDT', 'SOLUSDT', 'DARBTC', 'IRISBTC', 'RAREBRL', 'THETAETH', 'UTKBTC']


    try:
        # Read parquet files
        print(f"Reading files...")
        df1 = pd.read_parquet(file1_path)
        df2 = pd.read_parquet(file2_path)

        # # temp for debug
        # === this is critical for getting good result!!
        # df1 = df1[~df1['symbol'].isin(exclude_symbols)]
        # df1 = df1[df1['symbol'].str.endswith('USDT')]
        # df1.to_parquet ('improve_predictions_20241201.parquet')

        # Compare line numbers and truncate if necessary
        df1, df2 = compare_line_numbers(df1, df2)

        # Compare columns and get common columns
        common_columns = compare_columns(df1, df2)

        # Compare column types
        compare_column_types(df1, df2, common_columns)

        # Compare values for each common column
        print(
            f"\nComparing values in common columns (precision: {PRECISION} digits)..."
        )
        for column in sorted(common_columns):
            compare_column_values(df1, df2, column)

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
