import pandas as pd
import os

# Specify your data directory
data_dir = '/home/dp/backup_crypto/crypto/data/HFT/202401'

# Loop through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.csv') and filename.replace('.csv', '.parquet') not in os.listdir(data_dir):
        # Full path for input and output files
        csv_path = os.path.join(data_dir, filename)
        parquet_path = os.path.join(data_dir, filename.replace('.csv', '.parquet'))
        
        print(f"Converting {filename} to Parquet...")
        
        # Read CSV and write to Parquet
        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(parquet_path)
            print(f"Successfully converted {filename}")
            
            # Optional: Print file sizes for comparison
            csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
            parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # MB
            print(f"CSV size: {csv_size:.2f} MB")
            print(f"Parquet size: {parquet_size:.2f} MB")
            print(f"Compression ratio: {csv_size/parquet_size:.2f}x")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error converting {filename}: {str(e)}")

print("Conversion completed!")