"""
Dataset EDA Script for ArSarcasT

This script performs basic exploratory data analysis (EDA) on the
ArSarcasT dataset (Training & Test), including:

- Label distribution (%)
- Text length statistics (min, max, average)
- Summary per label and overall dataset

Outputs:
- Console summary
- CSV files: eda_train_summary.csv, eda_test_summary.csv
"""

import os
import sys
import pandas as pd


# Ensure UTF-8 output (useful for Arabic text in some terminals)
sys.stdout.reconfigure(encoding='utf-8')


# ------------------------------------------------------------------
# Paths Configuration
# ------------------------------------------------------------------
# Get the absolute path of the current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define dataset file paths
TRAIN_PATH = os.path.join(BASE_DIR, "Data", "ArSarcasT_Training.csv")
TEST_PATH = os.path.join(BASE_DIR, "Data", "ArSarcasT_Test.csv")


# ------------------------------------------------------------------
# EDA Function
# ------------------------------------------------------------------
def analyze_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Perform EDA on a dataset and return summary statistics.

    Parameters:
        df (pd.DataFrame): Input dataset containing 'TEXT' and 'LABEL' columns
        dataset_name (str): Name of the dataset (e.g., 'TRAIN', 'TEST')

    Returns:
        pd.DataFrame: Summary table with statistics per label and overall
    """

    # Create a new column for text length (character-based)
    df["TEXT_LEN"] = df["TEXT"].astype(str).apply(len)

    # --------------------------------------------------------------
    # Per-label summary statistics
    # --------------------------------------------------------------
    summary = (
        df.groupby("LABEL")
        .agg(
            Count=("TEXT", "count"),
            Max_Len=("TEXT_LEN", "max"),
            Min_Len=("TEXT_LEN", "min"),
            Avg_Len=("TEXT_LEN", "mean"),
        )
        .reset_index()
        .sort_values("LABEL")
    )

    # --------------------------------------------------------------
    # Label distribution (percentage)
    # --------------------------------------------------------------
    distribution = (
        df["LABEL"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str) + "%"
    )

    # Map distribution to summary table
    summary["Distribution"] = summary["LABEL"].map(distribution)

    # --------------------------------------------------------------
    # Overall dataset summary (all labels combined)
    # --------------------------------------------------------------
    overall = pd.DataFrame(
        {
            "LABEL": ["ALL"],
            "Count": [df["TEXT"].count()],
            "Max_Len": [df["TEXT_LEN"].max()],
            "Min_Len": [df["TEXT_LEN"].min()],
            "Avg_Len": [df["TEXT_LEN"].mean()],
            "Distribution": ["100%"],
        }
    )

    # Combine per-label and overall summaries
    final_summary = pd.concat([summary, overall], ignore_index=True)

    # Print results to console
    print(f"\n=== {dataset_name} Dataset Summary ===")
    print(final_summary)

    return final_summary


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------
if __name__ == "__main__":

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Run EDA analysis
    train_summary = analyze_dataset(train_df, "TRAIN")
    test_summary = analyze_dataset(test_df, "TEST")

    # Save results to CSV files (UTF-8 with BOM for Excel compatibility)
    train_output = os.path.join(BASE_DIR, "eda_train_summary.csv")
    test_output = os.path.join(BASE_DIR, "eda_test_summary.csv")

    train_summary.to_csv(train_output, index=False, encoding="utf-8-sig")
    test_summary.to_csv(test_output, index=False, encoding="utf-8-sig")

    print("\nEDA summaries saved successfully ✔")