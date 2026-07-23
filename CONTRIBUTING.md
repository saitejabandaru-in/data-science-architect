# Contributing to Data Science Architect

Thank you for your interest in contributing to **Data Science Architect**! We welcome contributions from data scientists, machine learning engineers, and AI developers to make machine learning workflows safer, more reproducible, and explainable for AI agents and human practitioners alike.

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### 1. Enhancing Skill Reference Manuals
You can improve or add new reference guides under `skills/data-science-architect/references/`. For example:
- Adding advanced feature selection strategies
- Deepening model evaluation protocols for time-series or multi-label classification
- Adding production monitoring and drift detection patterns

### 2. Improving Workflow Helper Scripts
Enhance execution scripts in `skills/data-science-architect/scripts/`:
- `workflow.py`: Add support for multi-target regression or hyperparameter tuning via Optuna.
- `validate_data.py`: Expand automated schema validation and statistical anomaly checks.

### 3. Adding New Examples
Add clear, self-contained examples under `examples/` demonstrating how to apply the 11-step workflow to real-world tasks.

## Getting Started

### Local Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/saitejabandaru-in/data-science-architect.git
   cd data-science-architect
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

5. **Test Workflow Script**
   ```bash
   python skills/data-science-architect/scripts/workflow.py --dataset examples/sample_dataset.csv
   ```

## Pull Request Guidelines

- Create a descriptive branch name (e.g., `feature/optuna-tuning` or `fix/leakage-check`).
- Ensure all unit tests pass with `pytest`.
- Keep code clean, type-hinted, and well-documented.
- Write or update tests covering any newly introduced features or scripts.
- Update relevant reference documentation if introducing new workflow steps or rules.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
