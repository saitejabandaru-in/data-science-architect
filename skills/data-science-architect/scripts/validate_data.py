#!/usr/bin/env python3
"""
Data Quality Validation Script - Data Science Architect Skill
Performs automated sanity audits on input tabular datasets.
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np


def validate_dataset(df: pd.DataFrame, target_col: str = None) -> dict:
    """Runs data quality checks on a DataFrame and returns audit results."""
    report = {
        "summary": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "duplicate_rows": int(df.duplicated().sum()),
        },
        "target_audit": {},
        "columns_audit": {},
        "warnings": []
    }

    # Duplicate rows warning
    if report["summary"]["duplicate_rows"] > 0:
        report["warnings"].append(f"Found {report['summary']['duplicate_rows']} duplicate rows.")

    # Target audit if specified
    if target_col:
        if target_col not in df.columns:
            report["warnings"].append(f"Target column '{target_col}' not found in dataset.")
        else:
            target_series = df[target_col]
            unique_vals = target_series.nunique()
            report["target_audit"] = {
                "name": target_col,
                "unique_values": unique_vals,
                "missing_count": int(target_series.isnull().sum()),
                "dtype": str(target_series.dtype)
            }
            if unique_vals <= 10:
                value_counts = target_series.value_counts(normalize=True).to_dict()
                report["target_audit"]["class_distribution"] = {str(k): round(v, 4) for k, v in value_counts.items()}
                min_ratio = min(value_counts.values())
                if min_ratio < 0.10:
                    report["warnings"].append(f"Severe class imbalance detected in target '{target_col}' (minority ratio: {min_ratio:.1%}).")

    # Column-level audit
    for col in df.columns:
        if col == target_col:
            continue

        series = df[col]
        null_count = int(series.isnull().sum())
        null_ratio = null_count / len(df)
        n_unique = series.nunique()
        dtype = str(series.dtype)

        col_info = {
            "dtype": dtype,
            "null_count": null_count,
            "null_ratio": round(null_ratio, 4),
            "n_unique": n_unique,
        }

        # Quality warnings
        if null_ratio > 0.50:
            report["warnings"].append(f"Column '{col}' has {null_ratio:.1%} missing values (consider dropping).")
        elif null_ratio > 0.10:
            report["warnings"].append(f"Column '{col}' has {null_ratio:.1%} missing values (imputation required).")

        if n_unique == 1:
            report["warnings"].append(f"Column '{col}' has zero variance (constant value).")
        elif n_unique == len(df) and dtype in ['object', 'category', 'string']:
            report["warnings"].append(f"Column '{col}' appears to be a unique identifier/ID column.")

        # Numeric stats & outlier audit
        if pd.api.types.is_numeric_dtype(series):
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            outliers = series[(series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))]
            col_info["min"] = float(series.min()) if not series.empty else None
            col_info["max"] = float(series.max()) if not series.empty else None
            col_info["mean"] = round(float(series.mean()), 4) if not series.empty else None
            col_info["outlier_count"] = len(outliers)
            if len(outliers) > 0 and len(df) > 0:
                outlier_ratio = len(outliers) / len(df)
                if outlier_ratio > 0.05:
                    report["warnings"].append(f"Column '{col}' has {outlier_ratio:.1%} potential outliers.")

        report["columns_audit"][col] = col_info

    return report


def main():
    parser = argparse.ArgumentParser(description="Data Quality Inspector - Data Science Architect")
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--target", default=None, help="Target column name (optional)")
    parser.add_argument("--json", action="store_true", help="Output report in JSON format")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataset)
    except Exception as e:
        print(f"Error loading dataset from {args.dataset}: {e}", file=sys.stderr)
        sys.exit(1)

    audit_report = validate_dataset(df, args.target)

    if args.json:
        print(json.dumps(audit_report, indent=2))
    else:
        print("\n" + "=" * 55)
        print("         DATA SCIENCE ARCHITECT - QUALITY REPORT")
        print("=" * 55)
        print(f"Dataset Path  : {args.dataset}")
        print(f"Total Rows    : {audit_report['summary']['total_rows']}")
        print(f"Total Columns : {audit_report['summary']['total_cols']}")
        print(f"Duplicates    : {audit_report['summary']['duplicate_rows']}")
        print(f"Memory Usage  : {audit_report['summary']['memory_usage_mb']} MB")

        if audit_report['warnings']:
            print("\n⚠️ WARNINGS DETECTED:")
            for w in audit_report['warnings']:
                print(f"  • {w}")
        else:
            print("\n✅ No data quality warnings detected!")

        print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()
