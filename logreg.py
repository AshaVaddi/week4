import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset from OpenML
boston = datasets.fetch_openml(name='boston', version=1)
df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

# Define features and target
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
