<div align="center">

# 🛒 E-Commerce Customer Behavior Analysis & Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

_An end-to-end machine learning project for analyzing customer behavior patterns and predicting churn in e-commerce platforms_

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Methodology](#-methodology) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Technologies Used](#-technologies-used)
- [Results & Insights](#-results--insights)
- [Dashboard](#-dashboard)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a comprehensive machine learning pipeline to analyze e-commerce customer behavior and predict customer churn. By leveraging advanced ML algorithms and data analysis techniques, businesses can:

- 📊 **Understand** customer purchasing patterns
- 🎯 **Segment** customers based on RFM (Recency, Frequency, Monetary) analysis
- ⚠️ **Predict** which customers are likely to churn
- 💡 **Optimize** marketing strategies and retention campaigns
- 📈 **Increase** customer lifetime value

---

## ✨ Features

### 🔍 Data Analysis

- Comprehensive exploratory data analysis (EDA)
- Statistical insights and correlation analysis
- Customer segmentation using RFM methodology
- Behavioral pattern identification

### 🤖 Machine Learning

- Multiple ML algorithms (Random Forest, XGBoost, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Model performance comparison

### 📊 Visualization

- Interactive Streamlit dashboard
- Real-time predictions
- Customer segment visualization
- Churn analysis by demographics

### 🛠️ Production-Ready Code

- Modular and reusable Python modules
- Comprehensive data preprocessing pipeline
- Feature engineering automation
- Model evaluation utilities

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/jackstealer/ECOMMERCE_CUSTOMER_BEHAVIOR.git
cd ECOMMERCE_CUSTOMER_BEHAVIOR
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🎬 Quick Start

### 1️⃣ Generate Synthetic Data

```bash
python src/data/generate_data.py
```

### 2️⃣ Run Jupyter Notebooks

Follow the notebooks in sequence:

```bash
jupyter notebook
```

- `01_data_understanding.ipynb` - Data exploration
- `02_data_cleaning.ipynb` - Data preprocessing
- `03_eda.ipynb` - Exploratory analysis
- `04_feature_engineering.ipynb` - Feature creation
- `05_modeling.ipynb` - Model training
- `06_evaluation.ipynb` - Model evaluation

### 3️⃣ Launch Interactive Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
ECOMMERCE_CUSTOMER_BEHAVIOR/
│
├── 📂 data/
│   ├── raw/                          # Original datasets (never edit)
│   ├── processed/                    # Cleaned & feature-engineered data
│   └── outputs/                      # Model predictions & recommendations
│
├── 📓 notebooks/
│   ├── 01_data_understanding.ipynb   # Phase 1: Data Collection & Understanding
│   ├── 02_data_cleaning.ipynb        # Phase 2: Cleaning & Preprocessing
│   ├── 03_eda.ipynb                  # Phase 3: Exploratory Data Analysis
│   ├── 04_feature_engineering.ipynb  # Phase 4: Feature Engineering
│   ├── 05_modeling.ipynb             # Phase 5: Model Development
│   └── 06_evaluation.ipynb           # Phase 6: Evaluation & Interpretation
│
├── 🐍 src/
│   ├── data/
│   │   ├── generate_data.py          # Synthetic dataset generator
│   │   └── preprocess.py             # Data cleaning functions
│   ├── features/
│   │   └── build_features.py         # Feature engineering pipeline
│   └── models/
│       ├── train.py                  # Model training scripts
│       └── evaluate.py               # Evaluation utilities
│
├── 📊 reports/
│   └── figures/                      # Saved charts and plots
│
├── 🎨 dashboard/
│   └── app.py                        # Streamlit dashboard (Phase 7)
│
├── 📄 requirements.txt               # Project dependencies
├── 📝 README.md                      # Project documentation
└── 🚫 .gitignore                     # Git ignore rules
```

---

## 🔬 Methodology

### Phase 1: Data Collection & Understanding

- Load and inspect raw customer data
- Understand data structure and types
- Identify data quality issues

### Phase 2: Data Cleaning & Preprocessing

- Handle missing values
- Remove duplicates
- Detect and treat outliers
- Data type conversions

### Phase 3: Exploratory Data Analysis

- Statistical analysis
- Distribution visualization
- Correlation analysis
- Customer behavior patterns

### Phase 4: Feature Engineering

- **RFM Analysis**: Recency, Frequency, Monetary scores
- **Engagement Metrics**: Email open rates, website visits
- **Derived Features**: Purchase per visit, CLV estimation
- **Encoding**: Categorical variable transformation

### Phase 5: Model Development

- Train/test split with stratification
- Multiple algorithm comparison:
  - Random Forest Classifier
  - XGBoost Classifier
  - Gradient Boosting Classifier
- Hyperparameter tuning
- Cross-validation

### Phase 6: Model Evaluation

- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix analysis
- ROC-AUC curve
- Feature importance visualization

### Phase 7: Dashboard Deployment

- Interactive Streamlit web application
- Real-time predictions
- Visual analytics

---

## 🛠️ Technologies Used

### Core Technologies

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Machine Learning

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)

### Visualization

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

### Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📈 Results & Insights

### Key Findings

- 🎯 **Churn Rate**: Comprehensive analysis of customer churn patterns
- 💰 **High-Value Customers**: Identification of top revenue contributors
- 📊 **RFM Segmentation**: Multiple distinct customer segments identified
- ⏰ **Critical Window**: Analysis of customer inactivity thresholds

### Model Performance

| Model             | Features                                  |
| ----------------- | ----------------------------------------- |
| Random Forest     | High interpretability, robust performance |
| XGBoost           | Best accuracy, handles imbalanced data    |
| Gradient Boosting | Strong generalization capabilities        |

---

## 🎨 Dashboard

The interactive Streamlit dashboard provides:

- 📊 **Overview Page**: Key metrics and KPIs
- 👥 **Customer Segmentation**: RFM analysis and clustering
- ⚠️ **Churn Analysis**: Risk factors and predictions
- 🔮 **Prediction Interface**: Real-time churn prediction for new customers

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Atul**

- GitHub: [@jackstealer](https://github.com/jackstealer)
- Repository: [ECOMMERCE_CUSTOMER_BEHAVIOR](https://github.com/jackstealer/ECOMMERCE_CUSTOMER_BEHAVIOR)

---

## 🙏 Acknowledgments

- Inspired by real-world e-commerce analytics challenges
- Built with modern ML best practices
- Designed for educational and commercial use

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ and ☕

</div>
