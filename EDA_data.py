import kagglehub
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import pickle

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
# Fix for displaying the minus sign '-' correctly in plots
plt.rcParams['axes.unicode_minus'] = False 

# --- 1. Create a directory for storing charts ---
# Define an output directory name
output_dir = "output_charts"
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建文件夹: {output_dir}")

# --- 2. Download the dataset ---
try:
    print("Downloading dataset from Kaggle Hub...")
    # kagglehub.dataset_download returns the path to the extracted dataset folder
    path = kagglehub.dataset_download("tencars/392-crypto-currency-pairs-at-minute-resolution")
    print(f"Dataset downloaded and extracted to: {path}")
except Exception as e:
    print(f"Failed to download dataset. Please check your internet connection and Kaggle API authentication. Error: {e}")
    path = "./tencars-392-crypto-currency-pairs-at-minute-resolution" 
    if not os.path.exists(path):
        print("Local dataset path not found, the script cannot continue.")
        exit()
    print(f"Using local cache path: {path}")


# --- 3. Define data loading and preprocessing function using DuckDB ---
def load_and_preprocess_with_duckdb(file_path, resample_rule='1h'):
    """
    Loads data from a CSV file using DuckDB, and performs resampling and preprocessing.
    """
    coin_name = os.path.basename(file_path).replace('.csv', '')
    print(f"\n>>>>> Starting to process with DuckDB: {coin_name.upper()} <<<<<")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # DuckDB SQL Query
    # Core idea:
    # 1. read_csv_auto('{file_path}'): Directly reads data from the CSV file.
    # 2. to_timestamp(time / 1000): Converts Unix timestamp (in milliseconds) to DuckDB's timestamp type.
    # 3. date_trunc('hour', ...): This is DuckDB's way to resample by hour. It truncates the time to the beginning of the hour.
    # 4. GROUP BY time_bucket: Groups rows by the truncated time bucket.
    # 5. arg_max/arg_min: A clever trick to get the first/last value within a group.
    #    - arg_max(close, to_timestamp(time / 1000)) gets the 'close' value from the row with the latest timestamp in the group (i.e., 'last').
    #    - arg_min(open, to_timestamp(time / 1000)) gets the 'open' value from the row with the earliest timestamp in the group (i.e., 'first').
    query = f"""
    SELECT
        date_trunc('hour', to_timestamp(time / 1000)) AS time_bucket,
        arg_min(open, to_timestamp(time / 1000)) AS open,
        max(high) AS high,
        min(low) AS low,
        arg_max(close, to_timestamp(time / 1000)) AS close,
        sum(volume) AS volume
    FROM read_csv_auto(
        '{file_path}', 
        header=true, 
        columns={{'time': 'BIGINT', 'open': 'DOUBLE', 'high': 'DOUBLE', 'low': 'DOUBLE', 'close': 'DOUBLE', 'volume': 'DOUBLE'}}
    )
    WHERE time > 0 -- A simple validity check
    GROUP BY time_bucket
    ORDER BY time_bucket;
    """
    
    print("Executing DuckDB query (handling millisecond timestamps)...")

    # Execute the query and convert the result directly to a Pandas DataFrame
    try:
        df_resampled = duckdb.query(query).to_df()
    except Exception as e:
        print(f"DuckDB query failed: {e}")
        return None

    # Set the time column as the index for easier subsequent operations
    df_resampled.set_index('time_bucket', inplace=True)
    
    print(f"Processing complete. Obtained {len(df_resampled)} hourly records.")
    
    return df_resampled

# --- 4. Batch load data for multiple cryptocurrencies ---
coins_to_analyze = {
    'BTC': "btcusd.csv",
    'ETH': "ethusd.csv",
    'LTC': "ltcusd.csv",
}

coin_data = {}

print("\n--- Starting batch loading and processing of multiple cryptocurrency data (using DuckDB) ---")
for symbol, filename in coins_to_analyze.items():
    file_path = os.path.join(path, filename)
    coin_data[symbol] = load_and_preprocess_with_duckdb(file_path, resample_rule='1h')

# Filter out any data that failed to load
coin_data = {k: v for k, v in coin_data.items() if v is not None}

print("\n--- All coins processed! ---")

# --- 5. Validation and Exploratory Data Analysis (EDA) ---
if coin_data: # Check if the dictionary is not empty
    print("\n--- Data preparation complete, starting exploratory analysis ---")
    
    # Loop through all successfully loaded coins
    for symbol, df in coin_data.items():
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_analysis = df.copy()

        print(f"\n========== Starting analysis for: {symbol} ==========")
        
        # 5.1 Single Asset Analysis
        # Plot price and volume charts
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Price chart
        ax1.plot(df_analysis.index, df_analysis['close'], label='Close Price', color='blue')
        ax1.set_title(f'{symbol}/USD Hourly Close Price (Processed by DuckDB)', fontsize=16)
        ax1.set_ylabel('Price (USD)')
        ax1.grid(True)
        ax1.legend(loc='upper left') 
        
        # Volume chart
        ax2.bar(df_analysis.index, df_analysis['volume'], label='Volume', color='grey', width=0.02)
        ax2.set_title(f'{symbol}/USD Hourly Volume', fontsize=16)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        ax2.legend(loc='upper left') 
        
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{symbol}_price_volume_chart.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")        
        plt.show()

        # 5.2 Analyze Price Returns
        # Operate on the current loop's DataFrame 'df_analysis'
        df_analysis['returns'] = df_analysis['close'].pct_change()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(df_analysis['returns'].dropna(), bins=100, kde=True)
        plt.title(f'{symbol} Hourly Returns Distribution', fontsize=16)
        plt.xlabel('Hourly Return')
        plt.ylabel('Frequency')
        save_path = os.path.join(output_dir, f'{symbol}_returns_distribution_chart.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
        plt.show()
        
        print(f"\n--- {symbol} Hourly Return Statistics ---")
        print(df_analysis['returns'].describe())

else:
    print("\nFailed to load any data, cannot proceed with analysis.")

# 5.3: Multi-Asset Comparative Analysis 
if len(coin_data) > 1:
    print("\n--- 5.3 Starting Multi-Asset Comparative Analysis ---")

    # Prepare a wide-format DataFrame containing all close prices
    close_prices_list = [df[['close']].rename(columns={'close': symbol}) for symbol, df in coin_data.items()]
    close_prices_wide = pd.concat(close_prices_list, axis=1)
    # Fill missing values, a common strategy is forward fill
    close_prices_wide.ffill(inplace=True)

    # 5.3.1 Compare Normalized Price Trends
    # Normalize using the first non-null value to handle potential missing data at the beginning
    normalized_prices = (close_prices_wide / close_prices_wide.bfill().iloc[0]) * 100
    
    plt.figure(figsize=(15, 8))
    for symbol in normalized_prices.columns:
        plt.plot(normalized_prices.index, normalized_prices[symbol], label=symbol)
    
    plt.title('Major Cryptocurrencies Normalized Price Trend Comparison', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Base = 100)')
    plt.legend()
    plt.grid(True)
    
    # Save the image to output_dir
    save_path = os.path.join(output_dir, 'normalized_price_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.show()

    # 5.3.2 Compare Returns Distribution
    returns_wide = close_prices_wide.pct_change()
    returns_long = returns_wide.melt(var_name='symbol', value_name='return')

    # Plot KDE chart
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=returns_long, x='return', hue='symbol', fill=True, alpha=0.2, common_norm=False)

    plt.title('Major Cryptocurrencies Hourly Returns Distribution Comparison', fontsize=16)
    plt.xlabel('Hourly Return')
    plt.ylabel('Density')
    plt.xlim(-0.1, 0.1) 
    
    # Save the image to output_dir
    save_path = os.path.join(output_dir, 'returns_distribution_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.show()

# --- 6. Multi-Asset Correlation Analysis ---
if len(coin_data) > 1:
    print("\n--- Starting Multi-Asset Correlation Analysis ---")
    
    union_queries = []
    for symbol, df in coin_data.items():
        file_path = os.path.join(path, coins_to_analyze[symbol])
        union_queries.append(f"""
        SELECT
            '{symbol}' as symbol,
            date_trunc('hour', to_timestamp(time / 1000)) AS time_bucket,
            arg_max(close, to_timestamp(time / 1000)) AS close
        FROM read_csv_auto(
            '{file_path}',
            header=true, 
            columns={{'time': 'BIGINT', 'open': 'DOUBLE', 'high': 'DOUBLE', 'low': 'DOUBLE', 'close': 'DOUBLE', 'volume': 'DOUBLE'}}
        )
        WHERE time > 0
        GROUP BY time_bucket
        """)
        
    full_query = " UNION ALL ".join(union_queries)
    
    # Execute the query
    all_closes_long = duckdb.query(full_query).to_df()

    # Pivot the long-format data to wide-format
    close_prices_wide = all_closes_long.pivot(index='time_bucket', columns='symbol', values='close')

    # Calculate returns and the correlation matrix
    correlation_matrix = close_prices_wide.pct_change().corr()

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Major Cryptocurrencies Hourly Return Correlation Matrix (Processed by DuckDB)', fontsize=16)
    
    # Save the image to output_dir
    save_path = os.path.join(output_dir, 'return_correlation_matrix_duckdb.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {save_path}")
    plt.show()
    
else:
    print("\nOnly one or no coins were loaded successfully, cannot compute correlation matrix.")

print("\nAnalysis process finished.")

if coin_data:
    # Define the filename for saving
    processed_data_path = "processed_coin_data.pkl"
    # 'wb' means open the file in write binary mode
    with open(processed_data_path, 'wb') as f:
        pickle.dump(coin_data, f)
        
    print(f"\nData processing complete. The `coin_data` dictionary has been saved to: {processed_data_path}")
    print("You can now run the next script for model training.")

print("\nEDA_data.py script finished running.")