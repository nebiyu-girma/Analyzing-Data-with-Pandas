# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better visuals
sns.set(style='whitegrid')

# Task 1: Load and Explore the Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

try:
    # Load dataset
    df = pd.read_csv(url, names=column_names)
    print("Dataset loaded successfully.\n")
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head(), '\n')
    
    # Check data types and missing values
    print("Data types:")
    print(df.dtypes, '\n')
    print("Missing values per column:")
    print(df.isnull().sum(), '\n')
    
    # Clean dataset (drop missing values if any)
    if df.isnull().sum().any():
        df.dropna(inplace=True)
        print("Missing values dropped.\n")
    else:
        print("No missing values found.\n")
        
except Exception as e:
    print(f"Error: {e}")
    # Exit if dataset loading fails
    raise SystemExit

# Task 2: Basic Data Analysis
print("Basic Statistics:")
print(df.describe(), '\n')

# Group by species and compute mean
species_group = df.groupby('class').mean()
print("Mean values by species:")
print(species_group, '\n')

# Observations from analysis
print("Key Observations:")
print("- Setosa has the smallest petal dimensions (mean petal_length = 1.46 cm).")
print("- Virginica has the largest petal dimensions (mean petal_length = 5.55 cm).\n")

# Task 3: Data Visualization
# Line Chart (Observation Index vs Sepal Length)
plt.figure(figsize=(10, 5))
df['sepal_length'].plot(kind='line', color='teal')
plt.title('Sepal Length Across Observations (Line Chart)')
plt.xlabel('Observation Index')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Bar Chart (Average Petal Length by Species)
plt.figure(figsize=(10, 5))
species_group['petal_length'].plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Average Petal Length by Species (Bar Chart)')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.show()

# Histogram (Sepal Width Distribution)
plt.figure(figsize=(10, 5))
df['sepal_width'].hist(bins=15, edgecolor='black', color='orchid')
plt.title('Distribution of Sepal Width (Histogram)')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot (Sepal Length vs Petal Length by Species)
plt.figure(figsize=(10, 5))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
plt.scatter(
    df['sepal_length'], 
    df['petal_length'], 
    c=df['class'].map(colors), 
    alpha=0.7,
    edgecolor='w'
)
plt.title('Sepal Length vs Petal Length by Species (Scatter Plot)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# Custom legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label=species, 
               markerfacecolor=color, markersize=10, markeredgecolor='k')
    for species, color in colors.items()
]
plt.legend(handles=legend_elements, loc='upper left')
plt.show()




#How to Run
#Install Dependencies:

#bash
#pip install pandas matplotlib seaborn
#Run the Code:

#Copy the code into a Jupyter notebook or Python script.

#Execute all cells or run the script. All plots will render automatically.

#Output Preview
#Terminal Output:

#Dataset loaded successfully.
#First 5 rows of the dataset:
#   sepal_length  sepal_width  petal_length  petal_width        class
#0           5.1          3.5           1.4          0.2  Iris-setosa
#1           4.9          3.0           1.4          0.2  Iris-setosa
#...
#Mean values by species:
#                sepal_length  sepal_width  petal_length  petal_width
#class                                                               
#Iris-setosa            5.006        3.418         1.464        0.244
#Iris-versicolor        5.936        2.770         4.260        1.326
#Iris-virginica         6.588        2.974         5.552        2.026


#Plots: Line, bar, histogram, and scatter plots will open in separate windows or inline in Jupyter.
