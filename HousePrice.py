import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
# Assuming a CSV file with housing data
df = pd.read_csv('housing_price_dataset.csv')

# Display the first few rows of the dataframe
print(df.head())

# Basic Data Exploration
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handling missing values (example: fill with median)
df = df.fillna(df.median())

# Visualize relationships
sns.pairplot(df)
plt.show()

# Feature Selection and Target Variable
X = df[['feature1', 'feature2', 'feature3']]  # replace with actual feature names
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Plotting actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
