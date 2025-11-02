import pandas as pd

def preprocess_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the Health Insurance dataset.
    Steps:
    - Encode binary columns (e.g. gender, fraud)
    - One-hot encode categorical columns
    - Convert income to numeric
    - Drop unnecessary ID/date columns
    - Convert boolean columns to integers
    """

    # --- 1️⃣ Encode binary columns ---
    binary_cols = ['PatientGender', 'ClaimLegitimacy']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                'F': 1, 'M': 0,
                'Fraud': 1, 'Legitimate': 0
            })

    # --- 2️⃣ One-hot encode categorical columns ---
    multiple_cols = [
        'DiagnosisCode', 'ProcedureCode', 'ProviderSpecialty', 'ClaimStatus',
        'PatientMaritalStatus', 'PatientEmploymentStatus', 'ProviderLocation',
        'ClaimType', 'ClaimSubmissionMethod'
    ]
    existing_cols = [col for col in multiple_cols if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    # --- 3️⃣ Convert numeric column ---
    if 'PatientIncome' in df.columns:
        df['PatientIncome'] = pd.to_numeric(df['PatientIncome'], errors='coerce')

    # --- 4️⃣ Drop irrelevant columns ---
    to_drop = ['ClaimID', 'PatientID', 'ProviderID', 'ClaimDate']
    df = df.drop(columns=[col for col in to_drop if col in df.columns], errors='ignore')

    # --- 5️⃣ Convert boolean columns to integers ---
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # --- 6️⃣ Fill missing numeric values (optional safeguard) ---
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
