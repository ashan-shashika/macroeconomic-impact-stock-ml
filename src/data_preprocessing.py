import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


# CONSTANTS

START_DATE = '1990-01-01'
TRAIN_END = '2019-12-31'   # Training:   up to end of 2019
TEST_START = '2020-01-01'   # Test:       2020 – 2025
TEST_END = '2025-12-31'

RETURN_COLS = [
    'SP500_Return',
    'Tech_Return',
    'Healthcare_Return',
    'Finance_Return',
    'Industrial_Return',
    'Energy_Return',
]

MACRO_COLS = [
    'CPI_Change',
    'Rate_Change',
    'GDP_Growth',
    'Unemp_Change',
    'USD_Change',
    'VIX_Change',
    'Credit_Spread',
]

KEEP_COLS = ['Date'] + RETURN_COLS + MACRO_COLS


# helper functions for each step of the pipeline, called by the main
# ``data_preprocess_pipeline`` function.

def _load_data(filepath: str):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {filepath}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def _calculate_log_returns(df: pd.DataFrame, price_cols: list, suffix: str = '_Return'):
    df = df.copy()
    for col in price_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue
        return_col = col.replace('_Close', '') + suffix
        df[return_col] = np.log(df[col] / df[col].shift(1)) * 100
        valid = df[return_col].dropna()
        print(f"  {return_col}: mean={valid.mean():.2f}%  std={valid.std():.2f}%  "
              f"min={valid.min():.2f}%  max={valid.max():.2f}%")
    return df


def _calculate_sector_returns(df: pd.DataFrame):
    price_cols = [
        'SP500_Close', 'SP500_IT_Close', 'SP500_Healthcare_Close',
        'SP500_Financials_Close', 'SP500_Industrials_Close', 'SP500_Energy_Close',
    ]
    df = _calculate_log_returns(df, price_cols)
    rename_map = {
        'SP500_Return':          'SP500_Return',
        'SP500_IT_Return':       'Tech_Return',
        'SP500_Healthcare_Return': 'Healthcare_Return',
        'SP500_Financials_Return': 'Finance_Return',
        'SP500_Industrials_Return': 'Industrial_Return',
        'SP500_Energy_Return':   'Energy_Return',
    }
    return df.rename(columns=rename_map)


def _process_macro_variables(df: pd.DataFrame):
    df = df.copy()
    print("\n" + "=" * 60)
    print("PROCESSING MACRO VARIABLES")
    print("=" * 60)

    gdp_missing = df['GDP'].isnull().sum()
    df['GDP'] = df['GDP'].interpolate(method='cubic')
    print(f"GDP: Filled {gdp_missing} missing values (quarterly to monthly)")

    diff_vars = {
        'CPI':           'CPI_Change',
        'Interest_Rate': 'Rate_Change',
        'Unemployment':  'Unemp_Change',
        'USD_Index':     'USD_Change',
        'VIX':           'VIX_Change',
    }
    for col, new_col in diff_vars.items():
        df[new_col] = df[col].diff()
        valid = df[new_col].dropna()
        print(f"  {new_col}: mean={valid.mean():.3f}  std={valid.std():.3f}")

    df['GDP_Growth'] = ((df['GDP'] / df['GDP'].shift(12)) - 1) * 100
    valid_gdp = df['GDP_Growth'].dropna()
    print(f"  GDP_Growth (YoY %): mean={valid_gdp.mean():.2f}%  std={valid_gdp.std():.2f}%")

    if 'Credit_Spread' not in df.columns:
        df['Credit_Spread'] = df['BAA'] - df['AAA']
        print("  Credit_Spread: Calculated as BAA − AAA")
    else:
        valid_spread = df['Credit_Spread'].dropna()
        print(f"  Credit_Spread: mean={valid_spread.mean():.3f}  std={valid_spread.std():.3f}")

    print("=" * 60)
    return df


def _clean_and_select(df: pd.DataFrame, drop_initial_rows: int = 12):
    df = df.copy()
    print("\n" + "=" * 60)
    print("CLEANING DATA")
    print("=" * 60)
    initial_rows = len(df)

    # Drop first N rows (YoY GDP needs 12 months of history)
    df = df.iloc[drop_initial_rows:].reset_index(drop=True)
    print(f"Dropped first {drop_initial_rows} rows")

    # Apply start-date filter
    df = df[df['Date'] >= START_DATE].reset_index(drop=True)
    print(f"Filtered to {START_DATE} onwards")

    # Keep only the columns we actually need
    existing_keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[existing_keep]

    # Drop rows missing key columns
    key_cols = ['SP500_Return'] + MACRO_COLS
    before = len(df)
    df = df.dropna(subset=key_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing key values")

    print(f"\nFinal dataset: {len(df)} rows  ({initial_rows - len(df)} removed)")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("=" * 60)
    return df


def _validate_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    all_cols = [c for c in KEEP_COLS if c != 'Date' and c in df.columns]
    missing = df[all_cols].isnull().sum()
    if missing.sum() == 0:
        print("No missing values")
    else:
        print("Missing values:\n", missing[missing > 0])

    print("\nExtreme return values (|return| > 10%):")
    for col in RETURN_COLS:
        if col not in df.columns:
            continue
        extremes = df[df[col].abs() > 10]
        if len(extremes):
            print(f"  {col}:")
            for _, row in extremes.iterrows():
                print(f"    {row['Date'].strftime('%Y-%m-%d')}: {row[col]:.2f}%")
        else:
            print(f"  {col}: none")

    inf_check = df[all_cols].isin([np.inf, -np.inf]).sum()
    if inf_check.sum() == 0:
        print("\n✓ No infinite values")
    else:
        print("\nInfinite values:\n", inf_check[inf_check > 0])

    print("=" * 60)


# Main functions for public API
def data_preprocess_pipeline(filepath: str, save_processed: bool = True):
    """
    Main preprocessing pipeline.

    Steps
    -----
    1. Load raw CSV
    2. Compute log returns for SP500 + 5 sector indices
    3. Derive macro features (first-differences, YoY GDP growth, credit spread)
    4. Clean data, apply START_DATE filter, keep only KEEP_COLS
    5. Validate
    6. Optionally save to processed_data.csv

    Parameters
    ----------
    filepath : str
        Path to raw CSV file.
    save_processed : bool, default True
        Save cleaned dataframe to ``processed_data.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns: Date, RETURN_COLS, MACRO_COLS.
    """
    print("\n" + "=" * 50)
    print(" " * 15 + "DATA PREPROCESSING PIPELINE" + " " * 15)
    print("=" * 50 + "\n")

    df = _load_data(filepath)
    df = _calculate_sector_returns(df)
    df = _process_macro_variables(df)
    df = _clean_and_select(df, drop_initial_rows=12)
    _validate_data(df)

    if save_processed:
        out = '../data/processed/processed_data.csv'
        df.to_csv(out, index=False)
        print(f"\nProcessed data saved {out}")

    print("\n✓ Pipeline complete!\n")
    return df


def add_lagged_variables(filepath: str, optimal_lags: dict):
    """
    Load a processed CSV, append lagged macro columns, overwrite the file.

    Parameters
    ----------
    filepath : str
        Path to the (already processed) CSV file.
    optimal_lags : dict
        Mapping of column name → lag in months.
        Columns with lag 0 are skipped (kept contemporaneous).

        Example::

            optimal_lags = {
                'CPI_Change':    2,
                'GDP_Growth':    3,
                'Unemp_Change':  2,
                'Credit_Spread': 4,
            }

    Returns
    -------
    pd.DataFrame
        Dataframe with new ``<col>_lag<n>`` columns added.
    """
    print("\n" + "=" * 60)
    print("ADDING LAGGED VARIABLES")
    print("=" * 60)

    df = pd.read_csv(filepath, parse_dates=['Date'])

    for var, lag in optimal_lags.items():
        if lag == 0:
            print(f"  ✓ {var}: no lag (contemporaneous)")
            continue
        if var not in df.columns:
            print(f"  ⚠ '{var}' not found in dataframe — skipping")
            continue

        lagged_col = f"{var}_lag{lag}"
        df[lagged_col] = df[var].shift(lag)
        valid = df[lagged_col].dropna()
        print(f"  ✓ {lagged_col}: lag={lag} month(s)  "
              f"μ={valid.mean():.3f}  σ={valid.std():.3f}  "
              f"({lag} obs lost at head)")

    # Drop rows that became NaN in any new lag column
    new_lag_cols = [f"{var}_lag{lag}" for var, lag in optimal_lags.items()
                    if lag > 0 and var in df.columns]
    before = len(df)
    df = df.dropna(subset=new_lag_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"\n  Dropped {dropped} rows with NaN lag values")

    df.to_csv(filepath, index=False)
    print(f"\nUpdated file saved {filepath}")
    print("=" * 60)
    return df, new_lag_cols


def data_split(df, feature_cols=None, target_col='SP500_Return'):
    """
    Split dataset into train and test sets using chronological order.

    Returns
    -------
    dict with keys:
        X_train, X_test       : np.ndarray  — Feature matrices
        y_train, y_test       : np.ndarray  — Target arrays
        dates_train,dates_test: pd.Series   — Corresponding dates
        feature_cols          : list        — Feature names used

    Example
    -------
    splits = data_split(df, MACRO_COLS)
    X_train, y_train = splits['X_train'], splits['y_train']

    """
    if feature_cols is None:
        feature_cols = MACRO_COLS

    train_mask = df['Date'] <= TRAIN_END
    test_mask = df['Date'] >= TEST_START

    splits = {}
    for name, mask in [('train', train_mask), ('test', test_mask)]:
        subset = df.loc[mask]
        splits[f'X_{name}'] = subset[feature_cols].values
        splits[f'y_{name}'] = subset[target_col].values
        splits[f'dates_{name}'] = subset['Date'].reset_index(drop=True)

        y = splits[f'y_{name}']
        d = splits[f'dates_{name}']
        print(f"\n  {name.capitalize():10s}  n={len(y):4d}  "
              f"{d.min().date()} to {d.max().date()}  "
              f"μ={y.mean():.2f}%  σ={y.std():.2f}%")

    splits['feature_cols'] = feature_cols
    print(f"\n  Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print("=" * 60)
    return splits
