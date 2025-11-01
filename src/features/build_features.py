

##It isolates all feature engineering logic — that is, how raw data is transformed into model-ready features.

import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    Used for encoding binary columns such as gender or fraud status.
    """
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # Gender mapping
    if valset == {"M", "F"}:
        return s.map({"M": 0, "F": 1}).astype("Int64")

    # Claim legitimacy mapping
    if valset == {"Legitimate", "Fraud"}:
        return s.map({"Legitimate": 0, "Fraud": 1}).astype("Int64")

    # Generic 2-category mapping
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # Non-binary columns remain unchanged
    return s


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for the Insurance dataset.
    - Encodes binary columns (gender, claim legitimacy)
    - One-hot encodes multi-category features
    - Converts income to numeric
    - Drops ID/date columns
    - Converts boolean to integers
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # STEP 1: Identify categorical and numeric columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # STEP 2: Split categorical columns by number of unique values
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    print(f"   🔢 Binary features: {binary_cols}")
    print(f"   🎨 Multi-category features: {multi_cols}")

    # STEP 3: Apply deterministic binary encoding
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")

    # STEP 4: One-hot encode multi-category columns
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} columns...")
        original_shape = df.shape
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features")

    # STEP 5: Convert PatientIncome to numeric
    if 'PatientIncome' in df.columns:
        df['PatientIncome'] = pd.to_numeric(df['PatientIncome'], errors='coerce')
        print("   💰 Converted 'PatientIncome' to numeric")

    # STEP 6: Drop ID/date columns
    to_drop = ['ClaimID', 'PatientID', 'ProviderID', 'ClaimDate']
    drop_existing = [col for col in to_drop if col in df.columns]
    if drop_existing:
        df.drop(columns=drop_existing, inplace=True)
        print(f"   🧹 Dropped columns: {drop_existing}")

    # STEP 7: Convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted boolean columns to int: {bool_cols}")

    # STEP 8: Ensure integer dtype for binary columns
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df
