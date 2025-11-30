# Loan Status Classifier Project

**CPSC 322 - Introduction to Artificial Intelligence**  
**Fall 2025**

A comprehensive machine learning project to predict loan status (good vs bad loans) using the Lending Club dataset. 
## Project Overview

This project classifies loans as either "Good" (Fully Paid/Current) or "Bad" (Charged Off/Late/Default) based on borrower and loan characteristics.



## Project Structure

```
Loan-Status-Classifier-Project/
├── mysklearn/                    # Custom machine learning implementations
│   ├── __init__.py
│   ├── myrandomforest.py         # Custom Random Forest classifier
│   └── mytree.py                 # Custom Decision Tree classifier
├── tests/                        # Unit tests (TDD approach)
│   └── test_myrandomforest.py
├── Project Proposal.ipynb        # Technical report and implementation
├── lc_loan_truncated.csv         # Dataset (180,000 rows, 74 attributes)
├── requirements.txt              # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── .dockerignore                  # Files to exclude from Docker build
├── run_tests.py                  # Test runner script
└── README.md                     # This file
```

## Dataset

- **Source**: [Lending Club Issued Loans (Kaggle)](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans/data?select=lc_loan.csv)
- **Size**: 180,000 rows, 74 attributes
- **Time Period**: 2007-2015
- **Target Variable**: `loan_status` (binary: Good vs Bad loans)

## Installation

### Option 1: Docker (Recommended)

The easiest way to run this project is using Docker:

1. **Using Docker Compose (Recommended)**:
```bash
# Build and start the container
docker-compose up --build

# The Jupyter Lab will be available at http://localhost:8888
```

2. **Using Docker directly**:
```bash
# Build the image
docker build -t loan-classifier .

# Run the container
docker run -p 8888:8888 -v $(pwd):/app loan-classifier
```

The Jupyter Lab interface will be available at `http://localhost:8888` with no password required.

### Option 2: Local Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook "Project Proposal.ipynb"
```

Or use Jupyter Lab:
```bash
jupyter lab "Project Proposal.ipynb"
```

## Usage

### Running with Docker

1. **Start the container**:
```bash
docker-compose up
```

2. **Access Jupyter Lab**: Open your browser and navigate to `http://localhost:8888`

3. **Open the notebook**: Click on `Project Proposal.ipynb` in the file browser

4. **Run all cells**: Use `Cell > Run All` or run cells sequentially

5. **Stop the container**: Press `Ctrl+C` in the terminal, then run:
```bash
docker-compose down
```

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

The project compares multiple machine learning algorithms and identifies the best-performing model based on accuracy and ROC-AUC score. The custom Random Forest implementation performs competitively compared to random forest.
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


