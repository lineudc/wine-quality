# Wine Quality Analysis

Modernized Python project for analyzing Wine Quality datasets (Red and White Vinho Verde). This project performs Exploratory Data Analysis (EDA), Feature Engineering, Principal Component Analysis (PCA), and Machine Learning modeling to predict wine quality.

## 🚀 Features

- **Data Loading & Cleaning**: Automated handling of red and white wine datasets.
- **EDA**: Comprehensive distribution plots, outlier analysis, and correlation matrices.
- **Comparative Analysis**: Statistical comparison between red and white wines.
- **Feature Engineering**: Creation of enological features (e.g., total acidity, sugar/alcohol ratio).
- **PCA**: Dimensionality reduction and visualization.
- **Machine Learning**: Training and evaluation of multiple regression models (Random Forest, Gradient Boosting, etc.).

## 🛠️ Installation

This project uses a `Makefile` to simplify setup and execution.

### Prerequisites
- Python 3.8+
- `pip`
- `venv` (standard library)

### Setup
To create a virtual environment and install dependencies:

```bash
make setup
```

## 🏃 Usage

### Run Analysis
To execute the full analysis pipeline:

```bash
make run
```

This will generate:
- Console output with statistical summaries and model results.
- Plots and visualizations in the `outputs/` directory.

### Run Tests
To run unit tests:

```bash
make test
```

### Clean
To remove temporary files and caches:

```bash
make clean
```

## 📂 Project Structure

```
wine-quality/
├── data/
│   ├── raw/                  # Original CSV datasets
│   └── processed/            # Intermediate data
├── docs/                     # Documentation and legacy files
├── outputs/                  # Generated plots and reports
├── src/                      # Source code
│   ├── config.py             # Configuration
│   ├── data_loader.py        # Data loading logic
│   ├── eda.py                # Exploratory Data Analysis
│   ├── features.py           # Feature Engineering
│   ├── models.py             # ML Models & PCA
│   ├── visualization.py      # Plotting functions
│   └── main.py               # Main execution script
├── tests/                    # Unit tests
├── Makefile                  # Automation commands
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## 📊 Outputs

The analysis generates several visualizations in the `outputs/` folder, including:
- Feature distributions
- Quality distribution (General, Red, White)
- Outlier boxplots
- Correlation matrices
- PCA Scree plots and Biplots

## 📝 Credits

Original dataset: Cortez et al., 2009 - UCI Wine Quality Dataset.
Refactored and modernized by [Your Name/Agent Name].
