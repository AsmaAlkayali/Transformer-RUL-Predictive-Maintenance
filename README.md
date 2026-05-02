# Transformer-RUL-Predictive-Maintenance
# Transformer-Based Predictive Maintenance (RUL Prediction)

## Overview
This project implements a Transformer-based deep learning model to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA C-MAPSS dataset.

**The goal is to support predictive maintenance by estimating how long a machine can operate before failure.
**


## Model Architecture
- Transformer Encoder for sequential learning
- Positional encoding for time-series structure
- Fully connected regression head
- Loss function: MSE / Smooth L1 Loss

---

## Dataset
NASA C-MAPSS Turbofan Engine Dataset (FD001):
- Multivariate time-series sensor data
- Engine cycle data until failure
- Operational settings and sensor readings

---

## Workflow
1. Data preprocessing (normalization, windowing)
2. Feature selection and scaling
3. Transformer model training
4. Evaluation using RMSE and MAE
5. RUL prediction on test sequences

---

## Results
- Transformer captures long-term dependencies in sensor data
- Performs better than baseline LSTM in sequence modeling
- Evaluated using RMSE metric

---

## Tech Stack
- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Repository Structure
