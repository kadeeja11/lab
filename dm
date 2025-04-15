#import statements
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

#load dataset
df = pd.read_csv("all_stocks_5yr.csv", parse_dates=['date'])
df = df[df['Name'] == 'AAPL']
df = df.sort_values('date')
df.set_index('date', inplace=True)

#plot close price
plt.figure(figsize=(12,6))
plt.plot(df['close'])
plt.title("Apple stock closing price")
plt.xlabel("date")
plt.ylabel("close price")
plt.grid(True)
plt.show()

#decomposition
decomposition = seasonal_decompose(df['close'], model='additive', period=252)
decomposition.plot()
plt.tight_layout()
plt.show()

#Moving average
plt.figure(figsize=(12,6))
plt.plot(df['close'], label='Original', alpha=0.4)
plt.plot(df['close'].rolling(window=5).mean(), label='Moving Avg (5)', linewidth=2)
plt.plot(df['close'].ewm(span=5).mean(), label='Exponential Moving Avg (span=5)', linewidth=2)
plt.legend()
plt.title("Moving Averages")
plt.xlabel('date')
plt.ylabel('close price')
plt.grid(True)
plt.tight_layout()
plt.show()

#ADF test for stationarity
result = adfuller(df['close'])
print(f"ADF Statistics : {result[0]}")
print(f"P-value : {result[1]}")

#differencing if not stationary
df['close_diff'] = df['close'].diff().dropna()

#plot ACF and PACF
fig, ax = plt.subplots(1,2,figsize=(16,4))
plot_acf(df['close_diff'].dropna(),ax=ax[0])
plot_pacf(df['close_diff'].dropna(), ax=ax[1])
plt.show()

#fit ARIMA Model
model_arima = ARIMA(df['close'],order=(5,1,0))
model_arima_fit = model_arima.fit()
print(model_arima_fit.summary())

#forecast next 6 periods
forecast = model_arima_fit.forecast(steps=6)
print("Forecast for next 6 periods :")
print(forecast)

#Holt Winters Model
model_hw = ExponentialSmoothing(df['close'], trend='add', seasonal='add', seasonal_periods=252)
model_hw_fit = model_hw.fit()
hw_forecast = model_hw_fit.forecast(6)

#plot actual vs forecast
plt.figure(figsize=(12,6))
plt.plot(df['close'], label='Actual')
plt.plot(hw_forecast.index, hw_forecast.values, label="Holt Winters Forecast", marker='o')
plt.title("Holt Winters Actual vs Forecast")
plt.xlabel("Date")
plt.ylabel("Close price")
plt.legend()
plt.grid(True)
plt.show()

#VAR Model - Multivariate
df_multi = df[['open', 'high', 'low', 'close', 'volume']].dropna()
df_diff = df_multi.diff().dropna()
var_model = VAR(df_diff)
var_fit = var_model.fit(maxlags=15, ic='aic')
forecast_input = df_diff.values[-var_fit.k_ar:]
forecast_var = var_fit.forecast(y=forecast_input, steps=6)

#Convert to Dataframe
forecast_df = pd.DataFrame(forecast_var, columns=df_diff.columns)
print("VAR Forecast - Differenced values")
print(forecast_df)

#Evaluation functions
def evaluate_forecast(actual,predicted):
    mae = mean_absolute_error(actual,predicted)
    rmse = math.sqrt(mean_squared_error(actual,predicted))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

actual_next_6 = df['close'].iloc[-6:]
print("ARIMA Forecast Evaluation:")
evaluate_forecast(actual_next_6, forecast)
print("Holt-Winters Forecast Evaluation:")
evaluate_forecast(actual_next_6, hw_forecast)

#statistical measures
def ts_statistics(ts):
    print(f"Mean : {ts.mean():.2f}")
    print(f"standard deviation : {ts.std():.2f}")
    print(f"Variance : {ts.var():.2f}")
    print(f"Skew : {ts.skew():.2f}")
    print(f"kurt : {ts.kurt():.2f}")
    
ts = df['close']
ts_statistics(ts)

plt.figure(figsize=(10,5))
sns.histplot(ts, kde=True)
plt.title("Distribution of time series ()Closing prices")
plt.xlabel("close price")
plt.ylabel("frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
