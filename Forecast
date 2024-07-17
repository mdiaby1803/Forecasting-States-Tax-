import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
file_path = r'C:\Users\meman\Downloads\Sales_Tax_Collections_by_State.csv'
df = pd.read_csv(file_path)

# Create a Date column by combining month and year
df['Date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['numeric month'].astype(str) + '-01')

# Aggregate data by Date
df_aggregated = df.groupby('Date').agg({'value': 'sum'}).reset_index()

# Set the Date as the index
df_aggregated.set_index('Date', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df_aggregated.index, df_aggregated['value'], label='Sales Tax Collections')
plt.title('Sales Tax Collections Over Time')
plt.ylabel('Revenue')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Split the data into training and testing sets
train = df_aggregated.iloc[:-12]
test = df_aggregated.iloc[-12:]

# Plot training and testing data
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['value'], label='Training Data')
plt.plot(test.index, test['value'], label='Testing Data', color='orange')
plt.title('Training and Testing Data')
plt.ylabel('Revenue')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Fit the auto ARIMA model
model = auto_arima(train['value'], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)

# Plot model diagnostics
model.plot_diagnostics(figsize=(15, 10))
plt.show()

# Make predictions
predictions = model.predict(n_periods=12)
predicted_mean = pd.Series(predictions, index=test.index)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['value'], label='Training Data')
plt.plot(test.index, test['value'], label='Actual Revenue')
plt.plot(predicted_mean.index, predicted_mean, label='Predicted Revenue', color='red')
plt.title('Sales Tax Revenue Forecast')
plt.ylabel('Revenue')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the mean squared error
mse = mean_squared_error(test['value'], predicted_mean)
print(f'Mean Squared Error: {mse}')

# Plot residuals
residuals = test['value'] - predicted_mean
plt.figure(figsize=(10, 6))
plt.plot(residuals.index, residuals, label='Residuals', color='purple')
plt.title('Residuals of the Forecast')
plt.ylabel('Residuals')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(df_aggregated['value'], model='multiplicative', period=12)

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
decomposition.observed.plot(ax=ax1, title='Observed')
decomposition.trend.plot(ax=ax2, title='Trend')
decomposition.seasonal.plot(ax=ax3, title='Seasonal')
decomposition.resid.plot(ax=ax4, title='Residual')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
