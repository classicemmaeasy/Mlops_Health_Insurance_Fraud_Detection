import pandas as pd

def preprocess_data(df):
    """
    Perform full preprocessing on the insurance dataset:
    - Encode binary columns (e.g., gender, fraud)
    - One-hot encode categorical columns
    - Convert income to numeric
    - Drop unnecessary ID/date columns
    - Convert boolean columns to integers

    Parameters:
    df (pd.DataFrame): The raw input DataFrame

    Returns:
    pd.DataFrame: The cleaned and transformed DataFrame
    """

    # 1️⃣ Encode binary columns
    binary_cols = ['PatientGender', 'ClaimLegitimacy']
    df[binary_cols] = df[binary_cols].replace({
        'F': 1, 'M': 0,
        'Fraud': 1, 'Legitimate': 0
    })

    # 2️⃣ One-hot encode categorical columns
    multiple_cols = [
        'DiagnosisCode', 'ProcedureCode', 'ProviderSpecialty', 'ClaimStatus',
        'PatientMaritalStatus', 'PatientEmploymentStatus', 'ProviderLocation',
        'ClaimType', 'ClaimSubmissionMethod'
    ]
    df = pd.get_dummies(df, columns=multiple_cols, drop_first=True)

    # 3️⃣ Convert numeric column
    df['PatientIncome'] = pd.to_numeric(df['PatientIncome'], errors='coerce')

    # 4️⃣ Drop irrelevant columns
    to_drop = ['ClaimID', 'PatientID', 'ProviderID', 'ClaimDate']
    df.drop(to_drop, axis=1, inplace=True)

    # 5️⃣ Convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df
