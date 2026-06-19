# Wellbore Geology Prediction

Roughly 10,000 horizontal wells are drilled worldwide every year, yet much of the drilling process still relies on manual interpretation by experts. These operations require immense technical precision, where even small deviations from the target zone can lead to significant resource waste. So this project aims to ease the process with ML algorithms.

## Project overview

This pipeline predicts True Vertical Thickness (TVT) for horizontal wells using XGBoost regression. It processes wellbore logs (e.g., Gamma Ray GR, spatial coordinates X/Y/Z, and measured depth MD) and integrates them with typewell data (reference wells in the similar area) to generate features for training and inference.

**Model Performance**: Achieves ~6.3 RMSE on training and ~8.4 RMSE on validation.

**Kaggle Leaderboard Perfomance**: 1054th out of 3275 (19.06.2026)

## Project structure:


src/

- data_loader.py – Pipeline for loading raw data from .csv files
- features.py – Generates rolling statistics, lag features, etc.
- model_utils.py – Utilities for training and validating models
- prepare_data.py – Final preparations for model training


Notebooks

- exploration.ipynb – Explores data patterns
- model_training.ipynb – Full data preprocessing and model training
- prepare_submission.ipynb – Final pipeline for Kaggle submission


Configuration

- requirements.txt
- .gitignore

## Technologies:

- Programming: Python, Jupyter
- ML: Scikit-learn, XGBoost
- Data worlflow: Pandas, Numpy
- Analyses: Scipy, Matplotlib