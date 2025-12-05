# Loan Status Classifier Project

**CPSC 322 - Data Science Algorithms**  
**Fall 2025**

A comprehensive machine learning project to predict loan status (good vs bad loans) using the Lending Club dataset. 
## Project Overview

This project classifies loans as either "Good" (Fully Paid) or "Bad" (Charged Off/Late/Default) based on borrower and loan characteristics.

## Dataset

- **Source**: [Lending Club Issued Loans (Kaggle)](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans/data?select=lc_loan.csv)
- **Size**: 180,000 rows, 74 attributes
- **Time Period**: 2007-2015
- **Target Variable**: `loan_status` (binary: Good vs Bad loans)

### Running the Notebook

Simply run all cells in `Project Proposal.ipynb` sequentially. The notebook includes:

1. **Introduction**: Dataset description and project overview
2. **Data Analysis**: Exploratory data analysis with visualizations
3. **Classification Results**: 
   - Custom Random Forest implementation
   - Parameter tuning (N, M, F)
   - Comparison with single decision tree
   - Feature selection and discretization
   - Model comparison and evaluation
4. **Conclusion**: Summary of findings and future improvements
5. **Acknowledgments**: Dataset sources and tool acknowledgments

### Running Unit Tests

To run the unit tests for the custom Random Forest implementation:

```bash
python -m pytest tests/test_myrandomforest.py -v
```

Or using unittest:

```bash
python -m unittest tests.test_myrandomforest -v
```

## Custom Random Forest Implementation

The `MyRandomForestClassifier` follows these specifications:

1. **Pre-processing**: Automatically splits data into 1/3 test set and 2/3 remainder set
2. **fit()**: 
   - Generates N decision trees using bootstrapping
   - Selects F random features at each node
   - Selects M most accurate trees based on validation performance
3. **predict()**: Uses majority voting with M selected trees

### Parameters

- `n_trees` (N): Number of trees to generate (default: 20)
- `n_selected_trees` (M): Number of best trees to select (default: 7)
- `n_features` (F): Number of random features at each node (default: 2)
- `max_depth`: Maximum depth of each tree (default: 10)
- `random_state`: Random seed for reproducibility


## Results

The project compares multiple machine learning algorithms and identifies the best-performing model based on accuracy and recall and includes a custom Random Forest implementation,

## Technical Report

The complete technical report is included in `Project Proposal.ipynb` with:
- Detailed data analysis with visualizations
- Implementation details for all classifiers
- Parameter tuning results and analysis
- Model comparison and evaluation
- Discussion of challenges and future improvements

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- imbalanced-learn >= 0.10.0
- jupyter >= 1.0.0

## Acknowledgments

- **Dataset**: Lending Club Issued Loans from Kaggle
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn, imbalanced-learn
- **Course**: CPSC-322-01 Data Science Algorithms (Fall 2025)


