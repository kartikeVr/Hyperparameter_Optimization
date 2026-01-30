

```
# Hyperparameter Optimization Framework

## Overview
This project implements a **custom hyperparameter optimization framework** that supports **Grid Search** and **Random Search** for machine learning models. The framework is designed to be modular, extensible, and compatible with **scikit-learn pipelines**, while providing **parallel execution**, **result logging**, and **unit test coverage**.

The goal of this project is to demonstrate an understanding of how hyperparameter search works internally, rather than relying solely on built-in utilities such as `GridSearchCV`.

---

## Key Features
- Grid search over hyperparameter combinations  
- Random search with configurable iteration count  
- Integration with `scikit-learn` pipelines  
- Parallel execution using multiprocessing / joblib-style abstraction  
- Logging of search results to **CSV** and **JSON** formats  
- Unit test suite for validating framework components  

---

## Project Structure

```

project_root/
│
├── hyperparam_optimized/
│   ├── **init**.py
│   ├── searcher.py              # GridSearch & RandomSearch logic
│   ├── parallel_process.py      # ParallelExecutor
│   └── logger.py                # CSV / JSON logging utilities
│
├── tests/
│   ├── **init**.py
│   ├── grid_test.py             # Grid search unit tests
│   ├── random_test.py           # Random search unit tests
│   └── log_test.py              # Logging unit tests
│
├── sample_logs/
│   ├── grid_results.csv         # Sample grid search output
│   └── random_results.json      # Sample random search output
│
├── notebook.ipynb               # Example usage notebook
├── README.md
├── EXECUTION_REPORT.md
└── pyproject.toml

````

---

## Framework Components

### 1. Grid Search
`GridSearch` generates the Cartesian product of all provided hyperparameter values.

- Exhaustively evaluates all parameter combinations
- Deterministic and reproducible
- Suitable for smaller search spaces

---

### 2. Random Search
`RandomSearch` samples random combinations from the parameter space.

- Configurable number of iterations (`n_iter`)
- Faster for large search spaces
- Supports reproducibility via `random_state`

---

### 3. Parallel Execution
`ParallelExecutor` executes model evaluations in parallel.

- Supports `n_jobs=-1` to use all available CPU cores
- Executes cross-validation folds for each parameter configuration
- Abstracted to remain framework-agnostic

---

### 4. Logging
Search results are logged for reproducibility and analysis.

- CSV logging for grid search
- JSON logging for random search
- Each record contains:
  - Hyperparameter configuration
  - Mean cross-validation score
  - Standard deviation across folds

Sample logs are available in the `sample_logs/` directory.

---

## Example Usage

A full working example is provided in `notebook.ipynb`. The following is a condensed version of the notebook's code.

### 1. Imports

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

from hyprparam_optimized.searcher import GridSearch, RandomSearch
from hyprparam_optimized.parallel_process import ParallelExecutor
from hyprparam_optimized.logger import log_to_csv, log_to_json
```

### 2. Data Loading and Preparation

```python
# Load the dataset
df = pd.read_csv(r"train_and_test2.csv")

# Data cleaning and preparation
df = df.drop(columns=['zero','zero.4','zero.15','zero.17','zero.18'])
df = df.drop_duplicates()
df = df.dropna()

# Define features (X) and target (y)
y = df['2urvived']
x = df.drop(columns='2urvived')

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 3. Grid Search Example

```python
# Define the hyperparameter space for Grid Search
param_space = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__max_depth': [3, 4]
}

# Create a scikit-learn pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier())
])

# Create a GridSearch object and generate parameter combinations
grid = GridSearch(param_space)
grid_params = grid.generate()

# Run the hyperparameter search in parallel
executor = ParallelExecutor(n_jobs=-1)
results = executor.run(
    estimator=pipeline,
    X=x_train,
    y=y_train,
    param_list=grid_params,
    cv=5,
    scoring="accuracy"
)

# Log the results to a CSV file
log_to_csv(results, "sample_logs/grid_results.csv")
```

### 4. Random Search Example

```python
# Define the hyperparameter space for Random Search
param_space_random = {
    'classifier__n_estimators': [10, 20, 50, 100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__max_depth': [2, 3, 4, 5]
}

# Create a RandomSearch object
random_search = RandomSearch(param_space_random, n_iter=10, random_state=42)
random_params = random_search.generate()

# Run the hyperparameter search in parallel
results_random = executor.run(
    estimator=pipeline,
    X=x_train,
    y=y_train,
    param_list=random_params,
    cv=5,
    scoring="accuracy"
)

# Log the results to a JSON file
log_to_json(results_random, "sample_logs/random_results.json")
```


---

## Sample Hyperparameter Search Logs

The project includes real output logs generated by the framework:

* `sample_logs/grid_results.csv`
* `sample_logs/random_results.json`

These files demonstrate how the framework records evaluated parameter combinations and their cross-validated performance.

---

## Unit Test Suite

Unit tests are implemented using **pytest** to validate core framework functionality.

### Tests Included

* Grid search combination generation
* Random search iteration control
* CSV logging functionality
* JSON logging functionality

### Running the Tests

From the project root:

```bash
pytest -v
```

All tests are expected to pass successfully.

---

## Execution Report

A test execution summary is provided in:

```
EXECUTION_REPORT.md
```

The report includes:

* Test framework used
* Number of tests executed
* Pass/fail summary
* Pytest execution output

---

## Design Decisions

* Framework logic is implemented independently of `GridSearchCV`
* Emphasis is placed on clarity, modularity, and correctness
* Parallelism and logging are treated as first-class features
* Unit tests focus on framework behavior, not model performance

---

## Conclusion

This project successfully demonstrates the design and implementation of a custom hyperparameter optimization framework with grid and random search, parallel execution, logging, and test coverage. The framework mirrors the core functionality of established tools while remaining transparent and extensible.

---

## How to Run the Project

1. Install dependencies
2. Run the example script or notebook
3. Execute unit tests
4. Review sample logs and execution report

---

## Author

Kartike Verma


