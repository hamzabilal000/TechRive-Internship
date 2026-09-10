"""
Week 1 - Task 1: NumPy & Pandas fundamentals.

Covers:
  * NumPy array operations (creation, vectorised math, reshaping, aggregation)
  * Building a small *synthetic* dataset with pandas
  * Data cleaning: detecting and handling missing values
  * groupby analysis

Run:
    python 01_numpy_pandas.py
"""

import numpy as np
import pandas as pd


def numpy_basics() -> None:
    """Demonstrate core NumPy array operations."""
    print("=" * 60)
    print("NUMPY ARRAY OPERATIONS")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    a = np.arange(1, 13)                 # 1..12
    matrix = a.reshape(3, 4)             # reshape to 3x4
    print("1-D array          :", a)
    print("Reshaped (3x4)     :\n", matrix)

    # Vectorised element-wise math (no Python loops).
    print("Element-wise square:", (a ** 2)[:6], "...")
    print("Column sums        :", matrix.sum(axis=0))
    print("Row means          :", matrix.mean(axis=1))

    # Boolean masking and fancy indexing.
    evens = a[a % 2 == 0]
    print("Even values        :", evens)

    # Linear-algebra style op: matrix * vector.
    vec = np.array([1, 0, -1, 2])
    print("matrix @ vec       :", matrix @ vec)

    # Random sampling + summary stats.
    sample = rng.normal(loc=50, scale=5, size=1000)
    print(f"Random sample -> mean={sample.mean():.2f}, std={sample.std():.2f}")


def make_synthetic_dataframe() -> pd.DataFrame:
    """Create a small synthetic 'employees' dataset with some missing values."""
    rng = np.random.default_rng(seed=7)
    n = 20
    departments = rng.choice(["Engineering", "Sales", "Marketing"], size=n)
    ages = rng.integers(22, 60, size=n)
    salaries = rng.normal(70_000, 15_000, size=n).round(-2)

    df = pd.DataFrame(
        {
            "employee_id": range(1, n + 1),
            "department": departments,
            "age": ages,
            "salary": salaries,
            "years_experience": rng.integers(0, 30, size=n),
        }
    )

    # Inject missing values so we have something to clean.
    df.loc[[2, 9, 15], "salary"] = np.nan
    df.loc[[5, 11], "age"] = np.nan
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and handle missing values."""
    print("\n" + "=" * 60)
    print("DATA CLEANING (missing values)")
    print("=" * 60)

    print("Missing values per column:")
    print(df.isna().sum().to_string())

    cleaned = df.copy()
    # Numeric imputation strategy:
    #   * salary  -> fill with the median (robust to outliers)
    #   * age     -> fill with the rounded mean
    salary_median = cleaned["salary"].median()
    age_mean = round(cleaned["age"].mean())
    cleaned["salary"] = cleaned["salary"].fillna(salary_median)
    cleaned["age"] = cleaned["age"].fillna(age_mean)

    print(f"\nImputed salary NaNs with median = {salary_median:,.0f}")
    print(f"Imputed age NaNs with mean     = {age_mean}")
    print("Remaining missing values:", int(cleaned.isna().sum().sum()))
    return cleaned


def groupby_analysis(df: pd.DataFrame) -> None:
    """Aggregate metrics per department."""
    print("\n" + "=" * 60)
    print("GROUPBY ANALYSIS (per department)")
    print("=" * 60)

    summary = (
        df.groupby("department")
        .agg(
            headcount=("employee_id", "count"),
            avg_age=("age", "mean"),
            avg_salary=("salary", "mean"),
            max_experience=("years_experience", "max"),
        )
        .round(1)
        .sort_values("avg_salary", ascending=False)
    )
    print(summary.to_string())

    top = summary["avg_salary"].idxmax()
    print(f"\nHighest average salary: {top} "
          f"({summary.loc[top, 'avg_salary']:,.0f})")


def main() -> None:
    numpy_basics()
    df = make_synthetic_dataframe()
    print("\nSynthetic dataset (head):")
    print(df.head().to_string(index=False))
    cleaned = data_cleaning(df)
    groupby_analysis(cleaned)


if __name__ == "__main__":
    main()
