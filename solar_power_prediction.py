import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Dataset (Replace with your actual data)
data = {'Temperature': [25, 28, 30, 22, 27, 32],
        'Humidity': [60, 55, 50, 65, 58, 48],
        'Irradiance': [500, 700, 800, 300, 600, 900],
        'Solar_Power': [100, 150, 200, 50, 120, 250]}

df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df[['Temperature', 'Humidity', 'Irradiance']]
y = df['Solar_Power']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Print model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)