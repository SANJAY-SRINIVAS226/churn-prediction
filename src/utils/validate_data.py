import logging
from typing import List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Comprehensive production data validation framework for the hybrid Telco Churn dataset."""
    print("🔍 Starting data validation engine...")
    failed_expectations = []

    # === SCHEMA VALIDATION ===
    print("   📋 Auditing columns and required schema...")
    
    # 🌟 FIXED: Changed to lowercase 'customerID' to match your actual file!
    if "customerID" not in df.columns:
        failed_expectations.append("Missing column: customerID")

    core_columns = [
        "gender",
        "Partner",
        "Dependents",
        "SeniorCitizen",
        "MultipleLines",
        "InternetService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure",
        "MonthlyCharges",
        "Churn",
        "Age",
        "SubscriptionType",
        "SupportTickets",
        "SatisfactionScore",
        "UsageHoursPerWeek",
    ]

    for col in core_columns:
        if col not in df.columns:
            failed_expectations.append(f"Missing column: {col}")

    # If critical schema features are missing, exit early to avoid crashes
    if failed_expectations:
        print(f"❌ Data validation FAILED: Missing structural fields. Details: {failed_expectations}")
        return False, failed_expectations

    # === NULL VALUE CHECKS ===
    print("   📈 Auditing for critical missing null structures...")
    critical_fields = [
        "customerID",  # 🌟 FIXED here as well
        "tenure",
        "MonthlyCharges",
        "Age",
        "SupportTickets",
        "SatisfactionScore",
    ]
    for field in critical_fields:
        null_count = df[field].isna().sum()
        if null_count > 0:
            failed_expectations.append(
                f"Null values detected in {field}: {null_count} rows"
            )

    # === CATEGORICAL SET VALIDATION ===
    print("   💼 Validating categorical value boundaries...")
    invalid_genders = df[~df["gender"].isin(["Male", "Female"])]
    if not invalid_genders.empty:
        failed_expectations.append(
            f"Invalid gender strings found in {len(invalid_genders)} rows"
        )

    invalid_churn = df[~df["Churn"].isin(["Yes", "No"])]
    if not invalid_churn.empty:
        failed_expectations.append(
            f"Invalid Churn values found in {len(invalid_churn)} rows"
        )

    invalid_subs = df[~df["SubscriptionType"].isin(["Basic", "Premium", "Enterprise"])]
    if not invalid_subs.empty:
        failed_expectations.append(
            f"Invalid SubscriptionType fields found in {len(invalid_subs)} rows"
        )

    invalid_contracts = df[
        ~df["Contract"].isin(["Month-to-month", "One year", "Two year"])
    ]
    if not invalid_contracts.empty:
        failed_expectations.append(
            f"Invalid Contract timelines found in {len(invalid_contracts)} rows"
        )

    # === RANGE VALIDATION ===
    print("   📊 Testing numerical range thresholds...")

    # Age constraints validation (Must stay between 18 and 100)
    out_of_bounds_age = df[(df["Age"] < 18) | (df["Age"] > 100)]
    if not out_of_bounds_age.empty:
        failed_expectations.append(
            f"Age parameters out of limits (18-100) in {len(out_of_bounds_age)} rows"
        )

    # Tenure bounds validation (0 to 120 months)
    out_of_bounds_tenure = df[(df["tenure"] < 0) | (df["tenure"] > 120)]
    if not out_of_bounds_tenure.empty:
        failed_expectations.append(
            f"Tenure limits broken (0-120) in {len(out_of_bounds_tenure)} rows"
        )

    # Satisfaction constraints (1 to 10 scale rating)
    out_of_bounds_sat = df[
        (df["SatisfactionScore"] < 1) | (df["SatisfactionScore"] > 10)
    ]
    if not out_of_bounds_sat.empty:
        failed_expectations.append(
            f"SatisfactionScore limits broken (1-10) in {len(out_of_bounds_sat)} rows"
        )

    # Usage limits validation (Max hours in a week is 168)
    out_of_bounds_hours = df[
        (df["UsageHoursPerWeek"] < 0) | (df["UsageHoursPerWeek"] > 168)
    ]
    if not out_of_bounds_hours.empty:
        failed_expectations.append(
            f"UsageHoursPerWeek out of valid limits (0-168) in {len(out_of_bounds_hours)} rows"
        )

    # === PROCESS RESULTS ===
    print("\n" + "=" * 50)
    if len(failed_expectations) == 0:
        print(f"✅ Data validation PASSED: All structural rules clear!")
        is_valid = True
    else:
        print(
            f"❌ Data validation FAILED: {len(failed_expectations)} validation rules breached!"
        )
        print(f"   Detailed Failures: {failed_expectations}")
        is_valid = False
    print("=" * 50)

    return is_valid, failed_expectations
