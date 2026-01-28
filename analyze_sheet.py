import pandas as pd
import sys
import warnings
warnings.filterwarnings('ignore')

FILE = "/home/user/BB-Python-Model/BB Forecast Baseline Outputs v1.xlsx"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 80)

xl = pd.ExcelFile(FILE, engine='openpyxl')
sheet_name = sys.argv[1]
df = xl.parse(sheet_name)

print(f"SHEET: '{sheet_name}' | Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns:")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}  (dtype: {df[col].dtype})")

# Show data
max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100
print(f"\n--- FIRST {min(max_rows, len(df))} ROWS ---")
print(df.head(max_rows).to_string(index=True))

if len(df) > max_rows:
    print(f"\n... ({len(df) - max_rows} more rows) ...")
    print(f"\n--- LAST 5 ROWS ---")
    print(df.tail(5).to_string(index=True))

# Numeric summary
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if numeric_cols:
    print(f"\n--- NUMERIC SUMMARY ---")
    print(df[numeric_cols].describe().to_string())

# Unique values for key columns
for col in df.columns:
    col_lower = str(col).lower()
    cat_keywords = ['segment', 'month', 'period', 'metric', 'type', 'category',
                    'stage', 'product', 'name', 'label', 'group', 'status',
                    'scenario', 'portfolio', 'methodology', 'bucket', 'rating',
                    'grade', 'class', 'description', 'item', 'rule', 'check',
                    'source', 'data', 'approach', 'method']
    is_key = any(kw in col_lower for kw in cat_keywords)
    if df[col].dtype == 'object' or is_key:
        nunique = df[col].nunique()
        if nunique <= 100:
            uvals = sorted(df[col].dropna().unique(), key=str)
            print(f"\n  Unique '{col}' ({nunique}): {uvals}")
        else:
            print(f"\n  Unique '{col}' ({nunique}): TOO MANY - first 30: {sorted(df[col].dropna().unique(), key=str)[:30]}")

# Date columns
for col in df.select_dtypes(include='datetime').columns:
    print(f"\n  Date range '{col}': {df[col].min()} to {df[col].max()}")
