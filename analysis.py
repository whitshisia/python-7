import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --------------------------
# Task 1: Load and Explore Dataset
# --------------------------
try:
    # Load iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Clean missing values (if any)
    df = df.dropna()

except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# --------------------------
# Task 2: Basic Data Analysis
# --------------------------
print("\nBasic statistics:")
print(df.describe())

# Group by species and calculate mean
group_means = df.groupby("target").mean()
print("\nMean values grouped by species (target):")
print(group_means)

# --------------------------
# Task 3: Data Visualization
# --------------------------
sns.set_style("whitegrid")

# Line chart (simulate time-series using index vs. sepal length)
plt.figure(figsize=(8, 5))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.savefig("line_chart.png")
plt.close()

# Bar chart (average petal length per species)
plt.figure(figsize=(8, 5))
sns.barplot(x="target", y="petal length (cm)", data=df, ci=None)
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
plt.ylabel("Average Petal Length (cm)")
plt.savefig("bar_chart.png")
plt.close()

# Histogram (distribution of sepal width)
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=15, edgecolor='black')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()

# Scatter plot (Sepal length vs Petal length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="target", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.savefig("scatter_plot.png")
plt.close()

print("\nAll plots have been saved as PNG files (line_chart.png, bar_chart.png, histogram.png, scatter_plot.png).")
