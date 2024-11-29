#EMPLOYEE_DATASET_ABDUL_RAFAY_197_LAB06

# home task whole completed code:
#ABDUL RAFAY / 2022F-BSE-197 / LAB 06 / HOME TASK:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
df = pd.read_csv("extended_employee_data RAFAY.csv")
print("ABDUL RAFAY / 2022F-BSE-197 / LAB 06 / HOME TASK:\n")
# Display the first few rows of the dataset (optional)
print(df.head())
# Select features and target variable
# Let's predict 'Monthly Salary ($)' based on 'Age', 'Years at Company', 'Performance Rating', 'Training Hours', and 'Satisfaction Level (%)'
X = df[['Age', 'Years at Company', 'Performance Rating', 'Training Hours', 'Satisfaction Level (%)']]
y = df['Monthly Salary ($)']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression model
model = LinearRegression()
# Fit the model
model.fit(X_train, y_train)
# Make predictions
predictions = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R² Score (Accuracy): {r2}")
# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Monthly Salary ($)')
plt.ylabel('Predicted Monthly Salary ($)')
plt.title('Actual vs Predicted Monthly Salaries')
plt.legend()
plt.grid()
plt.show()
