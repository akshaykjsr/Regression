import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("Student_Performance.csv")

# Rename columns to avoid spaces
df = df.rename(columns=lambda x: x.replace(" ", "_"))

# Handle categorical variables (convert "Yes" to 1 and "No" to 0)
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})

# Define X (features) and y (target variable)
y = df['Performance_Index']
X = df.drop(columns=['Performance_Index'])

# Standardize the numerical features (excluding categorical ones)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Add a constant (intercept) AFTER standardization
X_scaled = sm.add_constant(X_scaled)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit the multiple linear regression model
model = sm.OLS(y_train, X_train).fit()

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Display the model summary
print(model.summary())
print(f'Mean Squared Error: {mse}')

# Residual plot to check homoscedasticity
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values (ŷ)")
plt.ylabel("Residuals")
plt.title("Residual Plot for Homoscedasticity Check")
plt.show()

# Checking VIF for multicollinearity (excluding intercept)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_scaled.columns[1:]  # Excluding 'const'
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i+1) for i in range(len(X_scaled.columns)-1)]
print(vif_data)
