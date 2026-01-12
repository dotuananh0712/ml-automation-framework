#!/usr/bin/env python3
"""Generate test datasets for manual validation.

Usage:
    python tests/manual/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_classification_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate customer churn classification dataset."""
    data = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_samples + 1)],
        "age": np.random.randint(18, 75, n_samples),
        "tenure_months": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.round(np.random.uniform(20, 150, n_samples), 2),
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"],
            n_samples,
            p=[0.5, 0.3, 0.2],
        ),
        "payment_method": np.random.choice(
            ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
            n_samples,
        ),
    })

    data["total_charges"] = np.round(
        data["monthly_charges"] * data["tenure_months"], 2
    )

    # Churn probability based on features
    churn_prob = (
        0.1
        + 0.3 * (data["tenure_months"] < 12).astype(float)
        + 0.2 * (data["monthly_charges"] > 80).astype(float)
        + 0.2 * (data["contract_type"] == "Month-to-month").astype(float)
    )
    data["churn"] = (np.random.random(n_samples) < churn_prob).astype(int)

    return data


def generate_bad_data(base_data: pd.DataFrame) -> pd.DataFrame:
    """Generate dataset with validation issues."""
    bad_data = base_data.copy()

    # Add missing values
    bad_data.loc[0:10, "age"] = None

    # Add invalid negative values
    bad_data.loc[11:15, "age"] = -5

    # Add invalid category
    bad_data.loc[16:20, "contract_type"] = "Invalid"

    return bad_data


def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test datasets...")

    # Generate classification data
    churn_data = generate_classification_data(500)

    # Save training data
    train_path = output_dir / "churn_train.parquet"
    churn_data.to_parquet(train_path, index=False)
    print(f"Created: {train_path} ({len(churn_data)} rows)")

    # Save test data (subset)
    test_data = churn_data.iloc[:100]
    test_path = output_dir / "churn_test.parquet"
    test_data.to_parquet(test_path, index=False)
    print(f"Created: {test_path} ({len(test_data)} rows)")

    # Save bad data for validation testing
    bad_data = generate_bad_data(churn_data)
    bad_path = output_dir / "churn_bad_data.parquet"
    bad_data.to_parquet(bad_path, index=False)
    print(f"Created: {bad_path} ({len(bad_data)} rows)")

    # Also save CSV versions
    csv_path = output_dir / "churn_train.csv"
    churn_data.to_csv(csv_path, index=False)
    print(f"Created: {csv_path}")

    print("\nTest datasets generated successfully!")
    print(f"\nData statistics:")
    print(f"  Total samples: {len(churn_data)}")
    print(f"  Churn rate: {churn_data['churn'].mean():.1%}")
    print(f"  Features: {list(churn_data.columns)}")


if __name__ == "__main__":
    main()
