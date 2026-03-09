import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    # Read CSV and parse dates
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


def calculate_log_returns(df, price_cols, suffix='_Return'):
    # Log return formula: ln(today / yesterday) * 100
    df = df.copy()

    for col in price_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue

        return_col = col.replace('_Close', '') + suffix
        df[return_col] = np.log(df[col] / df[col].shift(1)) * 100

        valid = df[return_col].dropna()
        print(f"{return_col}: mean={valid.mean():.2f}%, std={valid.std():.2f}%, "
              f"min={valid.min():.2f}%, max={valid.max():.2f}%")

    return df


def calculate_sector_returns(df):
    # List of all sector price columns
    price_cols = [
        'SP500_Close',
        'SP500_IT_Close',
        'SP500_Healthcare_Close',
        'SP500_Financials_Close',
        'SP500_Industrials_Close',
        'SP500_Energy_Close'
    ]

    df = calculate_log_returns(df, price_cols)

    # Give them simpler names
    rename_map = {
        'SP500_Return': 'SP500_Return',
        'SP500_IT_Return': 'Tech_Return',
        'SP500_Healthcare_Return': 'Healthcare_Return',
        'SP500_Financials_Return': 'Finance_Return',
        'SP500_Industrials_Return': 'Industrial_Return',
        'SP500_Energy_Return': 'Energy_Return'
    }

    df = df.rename(columns=rename_map)
    return df


def process_macro_variables(df):
    df = df.copy()

    print("\n" + "="*60)
    print("PROCESSING MACRO VARIABLES")
    print("="*60)

    # Fill quarterly GDP gaps to monthly using cubic interpolation
    gdp_missing = df['GDP'].isnull().sum()
    df['GDP'] = df['GDP'].interpolate(method='cubic')
    print(f"GDP: Filled {gdp_missing} missing values (quarterly -> monthly)")

    # First difference = change from previous month
    diff_vars = {
        'CPI': 'CPI_Change',
        'Interest_Rate': 'Rate_Change',
        'Unemployment': 'Unemp_Change',
        'USD_Index': 'USD_Change',
        'VIX': 'VIX_Change'
    }

    for col, new_col in diff_vars.items():
        df[new_col] = df[col].diff()
        valid = df[new_col].dropna()
        print(f"{new_col}: mean={valid.mean():.3f}, std={valid.std():.3f}")

    # Year-over-year GDP growth in %
    df['GDP_Growth'] = ((df['GDP'] / df['GDP'].shift(12)) - 1) * 100
    valid_gdp = df['GDP_Growth'].dropna()
    print(f"GDP_Growth (YoY %): mean={valid_gdp.mean():.2f}%, std={valid_gdp.std():.2f}%")

    # Credit spread is already stationary, use as-is
    if 'Credit_Spread' in df.columns:
        valid_spread = df['Credit_Spread'].dropna()
        print(f"Credit_Spread: mean={valid_spread.mean():.3f}, std={valid_spread.std():.3f}")
    else:
        # Calculate it if missing
        df['Credit_Spread'] = df['BAA'] - df['AAA']
        print("Credit_Spread: Calculated as BAA - AAA")

    print("="*60 + "\n")
    return df


def clean_data(df, min_date=None, drop_initial_rows=12):
    df = df.copy()
    initial_rows = len(df)

    print("\n" + "="*60)
    print("CLEANING DATA")
    print("="*60)

    # Drop first N rows since YoY GDP needs 12 months of history
    if drop_initial_rows > 0:
        df = df.iloc[drop_initial_rows:].reset_index(drop=True)
        print(f"Dropped first {drop_initial_rows} rows")

    # Keep only rows from this date onwards
    if min_date:
        df = df[df['Date'] >= min_date].reset_index(drop=True)
        print(f"Filtered data from {min_date} onwards")

    # Remove rows that have missing values in key columns
    key_cols = ['SP500_Return', 'CPI_Change', 'Rate_Change', 'GDP_Growth']
    before = len(df)
    df = df.dropna(subset=key_cols)
    dropped = before - len(df)

    if dropped > 0:
        print(f"Dropped {dropped} rows with missing key values")

    print(f"\nFinal dataset: {len(df)} rows ({initial_rows - len(df)} removed)")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print("="*60 + "\n")

    return df


def validate_data(df, return_cols, macro_cols):
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)

    all_cols = return_cols + macro_cols

    # Check for missing values
    missing = df[all_cols].isnull().sum()
    if missing.sum() == 0:
        print("No missing values found")
    else:
        print("Missing values:")
        print(missing[missing > 0])

    # Show dates with unusually large returns (> 10%)
    print("\nExtreme Return Values (|return| > 10%):")
    for col in return_cols:
        if col in df.columns:
            extremes = df[df[col].abs() > 10]
            if len(extremes) > 0:
                print(f"\n  {col}:")
                for _, row in extremes.iterrows():
                    print(f"    {row['Date'].strftime('%Y-%m-%d')}: {row[col]:.2f}%")
            else:
                print(f"  {col}: None")

    # Basic stats summary
    print("\nReturn Statistics:")
    print(df[return_cols].describe().round(2))

    print("\nMacro Variable Statistics:")
    print(df[macro_cols].describe().round(3))

    # Check for infinite values
    inf_check = df[all_cols].isin([np.inf, -np.inf]).sum()
    if inf_check.sum() == 0:
        print("\nNo infinite values found")
    else:
        print("\nInfinite values detected:")
        print(inf_check[inf_check > 0])

    print("="*60 + "\n")


def split_data(df, train_end='2020-12-31', val_end='2023-12-31',
               feature_cols=None, target_col='SP500_Return'):
    # Default macro features if none provided
    if feature_cols is None:
        feature_cols = [
            'CPI_Change', 'Rate_Change', 'GDP_Growth',
            'Unemp_Change', 'USD_Change', 'VIX_Change', 'Credit_Spread'
        ]

    print("\n" + "="*60)
    print("SPLITTING DATA (TIME SERIES)")
    print("="*60)

    # Create time-based boolean masks
    train_mask = df['Date'] <= train_end
    val_mask = (df['Date'] > train_end) & (df['Date'] <= val_end)
    test_mask = df['Date'] > val_end

    # Split features (X) and target (y)
    X_train = df.loc[train_mask, feature_cols].values
    X_val   = df.loc[val_mask, feature_cols].values
    X_test  = df.loc[test_mask, feature_cols].values

    y_train = df.loc[train_mask, target_col].values
    y_val   = df.loc[val_mask, target_col].values
    y_test  = df.loc[test_mask, target_col].values

    # Save date ranges for reference
    train_dates = df.loc[train_mask, 'Date']
    val_dates   = df.loc[val_mask, 'Date']
    test_dates  = df.loc[test_mask, 'Date']

    # Print summary of each split
    for name, dates, y in [('Training', train_dates, y_train),
                            ('Validation', val_dates, y_val),
                            ('Test', test_dates, y_test)]:
        print(f"\n{name} Set:")
        print(f"  Period: {dates.min()} to {dates.max()}")
        print(f"  Samples: {len(y)}")
        print(f"  Target mean: {y.mean():.2f}%, std: {y.std():.2f}%")

    print(f"\nFeatures ({len(feature_cols)}): {', '.join(feature_cols)}")
    print("="*60 + "\n")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            train_dates, val_dates, test_dates)


def preprocess_pipeline(filepath, train_end='2020-12-31',
                        val_end='2023-12-31', save_processed=True):
    """Run all preprocessing steps in order"""
    print("\nPREPROCESSING PIPELINE\n")

    df = load_data(filepath)
    df = calculate_sector_returns(df)
    df = process_macro_variables(df)
    df = clean_data(df, min_date='1990-01-01', drop_initial_rows=12)

    return_cols = ['SP500_Return', 'Tech_Return', 'Healthcare_Return',
                   'Finance_Return', 'Industrial_Return', 'Energy_Return']
    macro_cols = ['CPI_Change', 'Rate_Change', 'GDP_Growth',
                  'Unemp_Change', 'USD_Change', 'VIX_Change', 'Credit_Spread']

    validate_data(df, return_cols, macro_cols)

    split_result = split_data(df, train_end=train_end, val_end=val_end)

    if save_processed:
        output_path = '../data/processed/processed_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}\n")

    print("Pipeline complete!\n")
    return (df,) + split_result


def get_sector_data(df, sector, feature_cols,
                    train_end='2020-12-31', val_end='2023-12-31'):
    """Get train/val/test splits for one specific sector"""
    target_col = f'{sector}_Return'

    if target_col not in df.columns:
        available = df.filter(like='_Return').columns.tolist()
        raise ValueError(f"Sector '{sector}' not found. Available: {available}")

    result = split_data(df, train_end, val_end, feature_cols, target_col)

    # Return only X and y arrays (skip dates)
    return result[:6]


def add_lagged_variables(df, lag_config):
    """
    Add lagged versions of macro variables based on CCF analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with macro variables
    lag_config : Dict[str, int]
        Dictionary mapping variable names to optimal lag periods
        Example: {'CPI_Change': 2, 'Rate_Change': 1, ...}
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added lagged variables
        
    Example:
    --------
    >>> lag_config = {'CPI_Change': 2, 'Rate_Change': 1}
    >>> df = add_lagged_variables(df, lag_config)
    >>> print(df.columns)
    # ['CPI_Change', 'CPI_Change_lag2', 'Rate_Change', 'Rate_Change_lag1', ...]
    """
    df = df.copy()
    
    print("\n" + "="*60)
    print("ADDING LAGGED VARIABLES (based on CCF analysis)")
    print("="*60)
    
    for var, lag in lag_config.items():
        if lag == 0:
            print(f"✓ {var}: No lag (contemporaneous)")
            continue
            
        if var not in df.columns:
            print(f"⚠ Warning: {var} not found in dataframe, skipping...")
            continue
        
        # Create lagged variable
        lagged_col = f"{var}_lag{lag}"
        df[lagged_col] = df[var].shift(lag)
        
        # Print statistics
        valid_vals = df[lagged_col].dropna()
        print(f"✓ {lagged_col}: Created ({lag} month lag), "
              f"μ={valid_vals.mean():.3f}, σ={valid_vals.std():.3f}, "
              f"lost {lag} observations")
    
    print("="*60 + "\n")
    
    return df


def preprocess_pipeline_with_lags(filepath: str,
                                  train_end: str = '2020-12-31',
                                  val_end: str = '2023-12-31',
                                  use_lags: bool = True,
                                  save_processed: bool = True):
    """
    Complete preprocessing pipeline WITH lagged variables.
    Parameters:
    -----------
    use_lags : bool, default=True
        Whether to include lagged macro variables (based on CCF analysis)
    Returns same as original preprocess_pipeline()
    """

    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "PREPROCESSING PIPELINE WITH LAGS" + " "*16 + "║")
    print("╚" + "="*58 + "╝\n")

    df = load_data(filepath)
    df = calculate_sector_returns(df)
    df = process_macro_variables(df)

    # Add lagged variables based on CCF analysis
    if use_lags:
        optimal_lags = {
            'CPI_Change': 2,
            'GDP_Growth': 3,
            'Unemp_Change': 2,
            'Credit_Spread': 4
        }
        df = add_lagged_variables(df, optimal_lags)

    # Clean data
    df = clean_data(df, min_date='1990-01-01', drop_initial_rows=12)

    # Validate
    return_cols = ['SP500_Return', 'Tech_Return', 'Healthcare_Return', 
                   'Finance_Return', 'Industrial_Return', 'Energy_Return']

    # Updated feature list with lags
    if use_lags:
        macro_cols = [
            'CPI_Change', 'CPI_Change_lag2',
            'Rate_Change',
            'GDP_Growth', 'GDP_Growth_lag3',
            'Unemp_Change', 'Unemp_Change_lag2',
            'USD_Change',  # No lag
            'VIX_Change',  # No lag
            'Credit_Spread', 'Credit_Spread_lag4'
        ]
    else:
        macro_cols = [
            'CPI_Change', 'Rate_Change',
            'GDP_Growth', 'Unemp_Change',
            'USD_Change', 'VIX_Change',
            'Credit_Spread'
            ]
    
    validate_data(df, return_cols, macro_cols)
    
    # Split data
    split_result = split_data(
        df, train_end=train_end,
        val_end=val_end,
        feature_cols=macro_cols
        )
    
    # Save processed data
    if save_processed:
        output_path = 'processed_data_with_lags.csv' if use_lags else 'processed_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}\n")
    
    print("Pipeline complete!\n")
    
    return (df,) + split_result


if __name__ == "__main__":
    print("Data Preprocessing Module")
    print("=" * 60)
    print("\nExample usage:")
    print(">>> from data_preprocessing import preprocess_pipeline")
    print(">>> result = preprocess_pipeline('main_data.csv')")
    print(">>> df, X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = result")