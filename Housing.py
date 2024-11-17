import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('housing.csv')

# Display the first few rows of the dataframe
print(df.head())

# Basic Data Exploration
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handling missing values (example: fill with median)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)

# Visualize relationships
sns.pairplot(df)
plt.show()

# Feature Selection and Target Variable
features = df.drop('Price', axis=1)  # All columns except 'Price'
target = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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
