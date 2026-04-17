import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Download Data
data = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01')

# Step 2: Clean Data
data = data[['Close']]
data.dropna(inplace=True)

# Step 3: Feature Engineering
data['MA7'] = data['Close'].rolling(7).mean()
data['MA30'] = data['Close'].rolling(30).mean()
data['Return'] = data['Close'].pct_change()

data.dropna(inplace=True)

# Step 4: Prepare Data
X = data[['MA7', 'MA30', 'Return']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 5: Train Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)

# Step 7: Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# Step 8: Plot
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("Bitcoin Price Prediction")
plt.savefig("prediction.png")
plt.show()

plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red")

plt.title("Actual vs Predicted Prices")
plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()
plt.show()
