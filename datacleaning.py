import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Sample data
data = {
    'Name': ['Sanjana', 'Bob', 'Chaithanya', 'Durga', 'Eva'],
    'Age': [25, None, 30, 35, 29],
    'Salary': [50000, 60000, None, 70000, 65000],
    'Department': ['HR', 'Finance', 'IT', 'IT', 'HR'],
    'Experience': [5, 7, 8, None, 6]
}


# Create DataFrame
df = pd.DataFrame(data)


print("Original DataFrame:")
print(df)


# Drop the 'Name' column
df.drop(columns=['Name'], inplace=True)


# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numeric values with column means


# Feature scaling
scaler = MinMaxScaler()
df[['Salary', 'Experience']] = scaler.fit_transform(df[['Salary', 'Experience']])


# Convert categorical column to dummy variables
df = pd.get_dummies(df, columns=['Department'], drop_first=True)


print("\nPreprocessed DataFrame:")
print(df)


# Additional info
df.info()
print(df.shape)
print(df.describe())
