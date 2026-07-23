#!/usr/bin/env python3
"""
Example: Housing Price Regression using Data Science Architect Workflow
"""

import os
import sys
import pandas as pd
import numpy as np

# Add skill script directory to path
script_dir = os.path.join(os.path.dirname(__file__), "..", "skills", "data-science-architect", "scripts")
sys.path.append(script_dir)

from workflow import run_data_science_workflow


def generate_housing_dataset(filepath: str, n_samples: int = 250):
    """Generates synthetic housing price dataset."""
    np.random.seed(42)
    sqft = np.random.randint(600, 4000, size=n_samples)
    bedrooms = np.random.randint(1, 6, size=n_samples)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n_samples)
    neighborhood = np.random.choice(['Suburbs', 'Downtown', 'Waterfront'], size=n_samples)

    # Base price calculation with noise
    price = (sqft * 150) + (bedrooms * 12000) + (bathrooms * 25000) + np.random.normal(0, 25000, n_samples)
    price = np.where(neighborhood == 'Waterfront', price * 1.4, price)

    df = pd.DataFrame({
        'house_id': [f'H-{100+i}' for i in range(n_samples)],
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'neighborhood': neighborhood,
        'price': np.round(price, 2)
    })
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic housing dataset at '{filepath}' with {n_samples} rows.")


def main():
    print("=== Data Science Architect - Housing Price Regression Example ===")
    dataset_path = os.path.join(os.path.dirname(__file__), "housing_dataset.csv")
    generate_housing_dataset(dataset_path)

    output_dir = os.path.join(os.path.dirname(__file__), "housing_outputs")
    run_data_science_workflow(
        dataset_path=dataset_path,
        target_col="price",
        output_dir=output_dir,
        problem_type="regression"
    )


if __name__ == "__main__":
    main()
