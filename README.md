# Sales Tax Collections Forecasting

This repository contains a Jupyter notebook for forecasting sales tax collections. The analysis includes loading and visualizing the data, fitting an auto ARIMA model, making predictions, and analyzing residuals and seasonal components. Additionally, a PDF document with the resulting graphs is included.

## Files

- **`forecasting.ipynb`**: Jupyter notebook containing the code for data analysis and forecasting.
- **`forecast.py`**: PDF document with the graphs generated from the analysis.

## Analysis Overview

1. **Data Loading and Initial Plot**:
   - Load the sales tax collection data, create a `Date` column, and aggregate the data by date.
   - Plot the sales tax collections over time.

2. **Training and Testing Data Split**:
   - Split the data into training and testing sets, with the last 12 months used for testing.
   - Visualize the distribution of the training and testing data.

3. **Auto ARIMA Model Fitting and Diagnostics**:
   - Fit an auto ARIMA model to the training data.
   - Plot model diagnostics to assess the model fit.

4. **Forecasting and Predictions**:
   - Make predictions for the next 12 months.
   - Plot the forecast alongside the actual revenue for comparison.
   - Calculate the mean squared error (MSE) to evaluate prediction accuracy.

5. **Residuals Analysis**:
   - Plot the residuals of the forecast to check for patterns or anomalies in the forecast errors.

6. **Seasonal Decomposition**:
   - Decompose the time series into observed, trend, seasonal, and residual components.
   - Plot the decomposition to analyze underlying patterns in the data.

## Requirements

To run the notebook, you need to have the following Python packages installed:

- pandas
- matplotlib
- pmdarima
- scikit-learn
- statsmodels

You can install these packages using pip:

```sh
pip install pandas matplotlib pmdarima scikit-learn statsmodels
