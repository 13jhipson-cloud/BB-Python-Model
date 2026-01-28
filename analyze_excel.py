import pandas as pd
import warnings
warnings.filterwarnings('ignore')

FILE = "/home/user/BB-Python-Model/BB Forecast Baseline Outputs v1.xlsx"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)

xl = pd.ExcelFile(FILE, engine='openpyxl')
print(f"FILE: {FILE}")
print(f"NUMBER OF SHEETS: {len(xl.sheet_names)}")
print(f"SHEET NAMES: {xl.sheet_names}")
print("=" * 120)

for sheet_name in xl.sheet_names:
    print(f"\n{'#' * 120}")
    print(f"### SHEET: '{sheet_name}'")
    print(f"{'#' * 120}")

    df = xl.parse(sheet_name)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn headers ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  [{i}] {col}  (dtype: {df[col].dtype})")

    print(f"\n--- DATA (first {min(100, len(df))} rows) ---")
    print(df.head(100).to_string(index=True))

    if len(df) > 100:
        print(f"\n... ({len(df) - 100} more rows not shown) ...")
        print(f"\n--- LAST 10 ROWS ---")
        print(df.tail(10).to_string(index=True))

    # Numeric summary
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        print(f"\n--- NUMERIC SUMMARY STATISTICS ---")
        print(df[numeric_cols].describe().to_string())

    # Categorical / key columns - show unique values
    cat_keywords = ['segment', 'month', 'period', 'metric', 'type', 'category',
                    'stage', 'product', 'name', 'label', 'group', 'status',
                    'scenario', 'portfolio', 'methodology', 'bucket', 'rating',
                    'grade', 'class', 'description', 'item']
    for col in df.columns:
        col_lower = str(col).lower()
        is_key = any(kw in col_lower for kw in cat_keywords)
        # Also show uniques for object/string cols with few unique values
        if df[col].dtype == 'object' or is_key:
            nunique = df[col].nunique()
            if nunique <= 80:
                uvals = sorted(df[col].dropna().unique(), key=str)
                print(f"\n  Unique values in '{col}' ({nunique} unique): {uvals}")
            else:
                print(f"\n  Unique values in '{col}' ({nunique} unique): [TOO MANY - showing first 30]")
                print(f"    {sorted(df[col].dropna().unique(), key=str)[:30]}")

    # Date columns
    date_cols = df.select_dtypes(include='datetime').columns.tolist()
    for col in date_cols:
        print(f"\n  Date range in '{col}': {df[col].min()} to {df[col].max()}")

    print(f"\n{'=' * 120}")

print("\n\nDONE - Full analysis complete.")
