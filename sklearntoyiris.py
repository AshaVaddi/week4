from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Convert to DataFrame for easier exploration
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Display basic information about the dataset
#\print("Iris Dataset Description:\n", iris.DESCR)
print("\nFirst 5 rows of the dataset:\n", iris_df.head())
print("\nBasic statistics:\n", iris_df.describe())
print("\nClass distribution:\n", iris_df['species'].value_counts())
