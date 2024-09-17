# Import necessary libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris.head())

# Check for missing values
print("\nMissing values in each column:")
print(iris.isnull().sum())

# Basic statistical summary
print("\nBasic statistical analysis:")
print(iris.describe())


# Calculate the correlation matrix
correlation_matrix = iris.corr(numeric_only=True)

# Print the correlation matrix
print("\nCorrelation matrix:")
print(correlation_matrix)

# Visualizations

# 1. Histogram for each feature
print("\nDisplaying histograms for each feature...")
iris.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# 2. Pairplot to show relationships between variables
print("\nDisplaying pairplot of the dataset...")
sns.pairplot(iris, hue='species')
plt.show()

# 3. Correlation Heatmap
print("\nDisplaying correlation heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Boxplot for feature variability
print("\nDisplaying boxplot of Sepal Length by Species...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=iris)
plt.show()
