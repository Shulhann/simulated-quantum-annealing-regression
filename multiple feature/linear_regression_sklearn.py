import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Features: [Weight, Gear Count, Brand Popularity]
X = np.array([
    [10, 18, 7],  # Bicycle 1
    [12, 21, 8],  # Bicycle 2
    [9, 15, 6],   # Bicycle 3
    [14, 20, 9],  # Bicycle 4
    [11, 22, 10]  # Bicycle 5
])

# Prices of the bicycles
y = np.array([500, 600, 450, 700, 650])  # Corresponding prices


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Get coefficients
coefficients = model.coef_
intercept = model.intercept_

print("Estimated coefficients:", coefficients)
print("Intercept:", intercept)


# Predicted values
y_pred = model.predict(X_scaled)

# R-squared score (coefficient of determination)
r2 = r2_score(y, y_pred)

# Mean Squared Error
mse = mean_squared_error(y, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

print("R-squared:", r2)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)