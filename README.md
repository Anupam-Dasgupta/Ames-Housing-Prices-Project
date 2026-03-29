# Ames Housing Price Prediction

End-to-end machine learning pipeline for predicting house prices on the Ames Housing dataset.  
Built as a clean, reproducible resume project with proper preprocessing, modeling, and inference scripts.

---

## Project Overview

This project implements a stacked regression pipeline to predict **SalePrice** using the Ames Housing dataset.

**Key highlights**

- Structured ML pipeline (train → save → predict)
- Feature engineering and preprocessing
- Stacked ensemble (XGBoost + Lasso)
- Reproducible predictions
- Clean project layout for production readiness
- Achieved test RMSE of 0.1298

---

## Project Structure

```
- AMES_HOUSE_PRICES/
│
├── data/ # raw and cleaned datasets
├── models/ # saved trained models (.pkl)
├── notebooks/ # EDA and experimentation
├── src/ # core pipeline code
│ ├── train.py
│ ├── predict.py
│ ├── preprocessing.py
│ └── feature_engineering.py
├── submissions/ # Kaggle submission files
├── README.md
└── Requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Anupam-Dasgupta/Ames-Housing-Prices-Project.git
cd Ames-Housing-Prices-Project
```

## Install dependencies

```bash
pip install -r Requirements.txt
```

## Training the model

```bash
python src/train.py
```

This will:

- preprocess training data
- train stacked model
- save model to models/

## Making predictions

```bash
python src/predict.py
```

This will:

- load saved model
- transform test data
- generate submission file in submissions/

## Models used

- XGBoost
- LASSO
- Stacking Regressor (meta-learner)

## Author

Anupam Dasgupta
  
