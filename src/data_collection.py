import pandas as pd
from pathlib import Path


def merge_monthly_csv_files(data_folder: str) -> pd.DataFrame:
    """
    Merge all *_monthly_data.csv files in a folder into one DataFrame.
    Columns: Date, [STOCK]_Open, [STOCK]_Close, [STOCK]_Volume
    """
    csv_files = list(Path(data_folder).glob("*_monthly_data.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No *_monthly_data.csv files found in {data_folder}")

    merged_df = None

    for file_path in csv_files:
        # Get stock name from filename e.g. "AAPL_monthly_data.csv" -> "AAPL"
        stock = file_path.stem.replace("_monthly_data", "")

        # Load file, keep only needed columns, rename with stock prefix
        df = pd.read_csv(file_path, parse_dates=["Date"])
        df = df[["Date", "Open", "Close", "Volume"]].rename(columns={
            "Open":   f"{stock}_Open",
            "Close":  f"{stock}_Close",
            "Volume": f"{stock}_Volume"
        })

        # Merge into main dataframe (outer join keeps all dates)
        merged_df = df if merged_df is None else pd.merge(merged_df, df, on="Date", how="outer")

    # Sort by date and forward-fill any missing values
    merged_df = merged_df.sort_values("Date").reset_index(drop=True).ffill()

    return merged_df
