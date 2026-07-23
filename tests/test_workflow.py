import os
import sys
import pandas as pd
import pytest

# Add scripts folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'skills', 'data-science-architect', 'scripts')))

from validate_data import validate_dataset
from workflow import run_data_science_workflow


@pytest.fixture
def sample_csv(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_data.csv"

    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'score': [10.5, 20.0, 15.2, 30.1, 25.4, 40.0, 35.5, 50.2, 45.1, 60.0],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    df.to_csv(p, index=False)
    return str(p)


def test_validate_dataset(sample_csv):
    df = pd.read_csv(sample_csv)
    report = validate_dataset(df, target_col='target')

    assert report['summary']['total_rows'] == 10
    assert report['summary']['total_cols'] == 5
    assert report['target_audit']['name'] == 'target'
    assert 'age' in report['columns_audit']


def test_workflow_classification(sample_csv, tmp_path):
    output_dir = tmp_path / "test_outputs"
    run_data_science_workflow(
        dataset_path=sample_csv,
        target_col='target',
        output_dir=str(output_dir),
        problem_type='classification'
    )

    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "evaluation_report.md").exists()
    assert (output_dir / "model_performance.png").exists()
