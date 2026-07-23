#!/usr/bin/env python3
"""
Example: Customer Churn Classification using Data Science Architect Workflow
"""

import sys
import os

# Add skill script directory to path
script_dir = os.path.join(os.path.dirname(__file__), "..", "skills", "data-science-architect", "scripts")
sys.path.append(script_dir)

from workflow import run_data_science_workflow
from validate_data import validate_dataset
import pandas as pd


def main():
    print("=== Data Science Architect - Customer Churn Example ===")
    dataset_path = os.path.join(os.path.dirname(__file__), "sample_dataset.csv")

    # Step 1: Pre-run Data Audit
    print("\nPhase 1: Running Data Quality Audit...")
    df = pd.read_csv(dataset_path)
    audit = validate_dataset(df, target_col="churn")
    print(f"Total Records: {audit['summary']['total_rows']} | Columns: {audit['summary']['total_cols']}")
    if audit['warnings']:
        print("Audit Warnings:")
        for w in audit['warnings']:
            print(f"  • {w}")

    # Step 2: Run End-to-End Workflow
    print("\nPhase 2: Executing End-to-End ML Pipeline...")
    output_dir = os.path.join(os.path.dirname(__file__), "churn_outputs")
    run_data_science_workflow(
        dataset_path=dataset_path,
        target_col="churn",
        output_dir=output_dir,
        problem_type="classification"
    )


if __name__ == "__main__":
    main()
