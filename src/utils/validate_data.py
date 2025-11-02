## validating schema integrity (all required columns exist).
## Check category validity (e.g. gender, claim type, etc.).
## Validate numeric ranges (e.g. PatientIncome should be ≥ 0).
## Confirm logical relationships (if possible).

import great_expectations as ge
from typing import Tuple, List

def validate_insurance_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Health Insurance dataset using Great Expectations.
    
    This function ensures schema correctness, data integrity, and logical validity
    before model training or feature engineering.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # Convert to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)




    # === SCHEMA VALIDATION ===
    print("   📋 Validating schema and required columns...")
    required_cols = [
        "ClaimID", "PatientID", "ProviderID", "ClaimDate", "ClaimLegitimacy",
        "PatientGender", "DiagnosisCode", "ProcedureCode", "ProviderSpecialty",
        "ClaimStatus", "PatientMaritalStatus", "PatientEmploymentStatus",
        "ProviderLocation", "ClaimType", "ClaimSubmissionMethod", "PatientIncome"
    ]
    for col in required_cols:
        ge_df.expect_column_to_exist(col)
        ge_df.expect_column_values_to_not_be_null(col)

    # === CATEGORY VALIDATION ===
    print("   🧩 Validating categorical values...")

    # Gender
    ge_df.expect_column_values_to_be_in_set("PatientGender", ["M", "F"])
    # Claim legitimacy
    ge_df.expect_column_values_to_be_in_set("ClaimLegitimacy", ["Fraud", "Legitimate"])

    # Marital and Employment
    ge_df.expect_column_values_to_be_in_set("PatientMaritalStatus", ["Single", "Married", "Divorced", "Widowed", "Other"])
    ge_df.expect_column_values_to_be_in_set("PatientEmploymentStatus", ["Employed", "Unemployed", "Self-employed", "Retired", "Student"])

    # Claim type and submission method
    ge_df.expect_column_values_to_be_in_set("ClaimType", ["Inpatient", "Outpatient", "Emergency", "Pharmacy"])
    ge_df.expect_column_values_to_be_in_set("ClaimSubmissionMethod", ["Online", "Paper", "Email", "Other"])

    # Claim status
    ge_df.expect_column_values_to_be_in_set("ClaimStatus", ["Approved", "Rejected", "Pending", "Investigating"])

    # === NUMERIC VALIDATION ===
    print("   📊 Validating numeric columns...")
    ge_df.expect_column_values_to_be_between("PatientIncome", min_value=0)
    ge_df.expect_column_values_to_not_be_null("PatientIncome")

    # === LOGICAL / CONSISTENCY CHECKS ===
    print("   🧠 Validating logical consistency...")
    # Example: Fraud claims might often have "Investigating" status
    if "ClaimStatus" in df.columns and "ClaimLegitimacy" in df.columns:
        ge_df.expect_column_pair_values_to_be_in_set(
            column_A="ClaimLegitimacy",
            column_B="ClaimStatus",
            value_pairs=[
                ("Fraud", "Investigating"),
                ("Fraud", "Rejected"),
                ("Legitimate", "Approved"),
                ("Legitimate", "Pending")
            ]
        )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️ Running validation checks...")
    results = ge_df.validate()

    failed_expectations = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"] if not r["success"]
    ]

    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks

    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results["success"], failed_expectations
