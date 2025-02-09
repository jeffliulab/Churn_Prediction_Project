# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to identify credit card customers that are most likely to churn. The project includes a Python package for a machine learning model that predicts customer churn, following coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The project involves:
- Loading and preprocessing customer data
- Performing EDA (Exploratory Data Analysis)
- Feature engineering and selection
- Model training (Random Forest and Logistic Regression)
- Model evaluation and results visualization

## Files and Data Description

### Main Files:
- `churn_library.py`: Contains the main functions for the machine learning pipeline
- `churn_script_logging_and_tests.py`: Contains unit tests for the functions in churn_library.py
- `README.md`: Project documentation
- `requirements_py3.6.txt`: Required Python packages
- `churn_notebook.ipynb`: Original notebook containing the solution (for reference)

### Directories:
- `/data`: Contains the bank customer data (bank_data.csv)
- `/images`: Stores EDA and results plots
  - `eda/`: Contains EDA visualizations
  - `results/`: Contains model performance plots
- `/logs`: Stores logging info and error messages
- `/models`: Stores trained model files

### Data:
The dataset (bank_data.csv) contains customer information including:
- Demographic info (age, gender, etc.)
- Bank relationship info (credit limit, card type, etc.)
- Transaction patterns
- Churn status (target variable)

## Running Files

### Environment Setup
```bash
python -m pip install -r requirements_py3.6.txt
```

### Running the Analysis
To run the full analysis:
```bash
python churn_library.py
```
This will:
1. Load and process the data
2. Perform EDA and save visualizations
3. Train and evaluate models
4. Save results and models

### Running Tests
To run the test suite:
```bash
python churn_script_logging_and_tests.py
```
This will:
1. Test all functions in churn_library.py
2. Log successes and errors to /logs/churn_library.log
3. Display test progress in the console

### Code Quality
To check code quality:
```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

To format code according to PEP 8:
```bash
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
```

### Expected Outputs
After running the full analysis:
- EDA plots will be saved in /images/eda/
- Model results will be saved in /images/results/
- Trained models will be saved in /models/
- Logs will be created in /logs/

After running tests:
- Test results will be displayed in the console
- Detailed logs will be saved in /logs/churn_library.log