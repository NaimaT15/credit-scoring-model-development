import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    """Loads the dataset from the given file path and returns a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path and try again.")
    return pd.read_csv(file_path)


# Function to display the first few rows
def display_head(data, n=5):
    """Displays the first 'n' rows of the DataFrame."""
    return data.head(n)

# Function to get the shape of the DataFrame
def get_shape(data):
    """Returns the number of rows and columns in the DataFrame."""
    return data.shape

# Function to get information about the data types and missing values
def get_info(data):
    """Prints the information of the DataFrame including data types and missing values."""
    return data.info()

# Function to get summary statistics of numerical columns
def get_summary_statistics(data):
    """Returns summary statistics for the numerical columns."""
    return data.describe()
def calculate_central_tendency_dispersion(data):
    """
    Calculates and returns a DataFrame with central tendency and dispersion measures for numerical columns.
    Includes mean, median, mode, standard deviation, variance, range, and IQR.
    """
    numeric_data = data.select_dtypes(include='number')  # Select only numerical columns
    stats = pd.DataFrame(index=numeric_data.columns)
    
    # Calculate central tendency metrics
    stats['Mean'] = numeric_data.mean()
    stats['Median'] = numeric_data.median()
    stats['Mode'] = numeric_data.mode().iloc[0]  # Mode returns a DataFrame, we select the first row
    
    # Calculate dispersion metrics
    stats['Standard Deviation'] = numeric_data.std()
    stats['Variance'] = numeric_data.var()
    stats['Range'] = numeric_data.max() - numeric_data.min()
    stats['IQR (Interquartile Range)'] = numeric_data.quantile(0.75) - numeric_data.quantile(0.25)
    
    return stats
def plot_numerical_distributions(data):
    """Generates histograms and boxplots for numerical features to visualize distributions, skewness, and outliers."""
    numeric_data = data.select_dtypes(include='number')  # Select only numerical columns
    columns = numeric_data.columns

    # Create a plot for each numerical column
    for col in columns:
        plt.figure(figsize=(14, 6))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(numeric_data[col], kde=True, color='skyblue')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=numeric_data[col], color='lightcoral')
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()
def plot_categorical_distributions(data, top_n=10):
    """
    Generates bar plots for each categorical feature to visualize category frequencies.
    If the number of unique categories in a column is too large, only the top N categories are displayed.
    
    Parameters:
    - data: DataFrame containing the dataset
    - top_n: Number of top categories to display for high cardinality columns (default: 10)
    """
    # Select only categorical columns
    categorical_data = data.select_dtypes(include='object')
    columns = categorical_data.columns

    # Create a bar plot for each categorical column
    for col in columns:
        plt.figure(figsize=(12, 6))
        
        # Count frequency of each category
        category_counts = categorical_data[col].value_counts()
        
        # Check if the column has high cardinality
        if len(category_counts) > top_n:
            # Display only the top N categories
            category_counts = category_counts.head(top_n)
            sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
            plt.title(f'Top {top_n} Categories in {col}')
        else:
            sns.countplot(x=categorical_data[col], palette='viridis', order=category_counts.index)
            plt.title(f'Distribution of {col}')
        
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')  # Rotate labels for better visibility
        plt.tight_layout()
        plt.show()
def correlation_analysis(data):
    """
    Calculates and visualizes the correlation matrix between numerical features.

    """
    # Select only numerical columns
    numeric_data = data.select_dtypes(include='number')
    
    # Calculate the correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Display the correlation matrix
    print("Correlation Matrix:\n", correlation_matrix)
    
    # Plot a heatmap of the correlation matrix
    plt.figure(figsize=(14, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()
def identify_missing_values(data):
    """
    Identifies and visualizes missing values in the dataset.
 
    """
    # Calculate the number and percentage of missing values for each column
    missing_data = data.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Values']
    missing_data['% of Total Values'] = 100 * missing_data['Missing Values'] / len(data)
    
    # Filter columns with missing values greater than 0
    missing_data = missing_data[missing_data['Missing Values'] > 0]
    
    # Sort the missing values in descending order
    missing_data = missing_data.sort_values(by='Missing Values', ascending=False).reset_index(drop=True)
    
    # Display missing value counts and percentages
    print("Columns with Missing Values:\n", missing_data)
    
    # Visualize the missing values using a heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap of Missing Values')
    plt.show()
    
    return missing_data
def detect_outliers(data):
  
    # Select only numerical columns
    numeric_data = data.select_dtypes(include='number')
    columns = numeric_data.columns

    # Iterate through each numerical column to create a box plot
    for col in columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=numeric_data[col], palette='viridis')
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.show()
        
        # Calculate and display IQR-based outliers
        Q1 = numeric_data[col].quantile(0.25)  # 25th percentile (Q1)
        Q3 = numeric_data[col].quantile(0.75)  # 75th percentile (Q3)
        IQR = Q3 - Q1  # Interquartile Range
        
        # Calculate the boundaries for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Display the number of outliers
        outliers = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)]
        print(f"{col}: {len(outliers)} outliers detected.")
        print(f"Outliers range below {lower_bound:.2f} or above {upper_bound:.2f}.\n")

def fill_outliers(data, columns=None):
 
    # Select only numerical columns if no specific columns are provided
    if columns is None:
        columns = data.select_dtypes(include='number').columns
    
    for col in columns:
        # Calculate the IQR for each column
        Q1 = data[col].quantile(0.25)  # 25th percentile (Q1)
        Q3 = data[col].quantile(0.75)  # 75th percentile (Q3)
        IQR = Q3 - Q1  # Interquartile Range

        # Calculate the lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap (replace) outliers to the lower and upper bounds
        data[col] = data[col].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))

        print(f"Column '{col}': Outliers filled using capping method with lower bound = {lower_bound:.2f} and upper bound = {upper_bound:.2f}.")
    
    return data

def create_advanced_aggregate_features(data):
 
    # Ensure the dataset has the necessary columns
    if 'CustomerId' not in data.columns or 'Amount' not in data.columns or 'ProductCategory' not in data.columns:
        raise ValueError("Dataset must contain 'CustomerId', 'Amount', and 'ProductCategory' columns")

    # Group by 'CustomerId' and calculate advanced aggregate metrics
    aggregate_features = data.groupby('CustomerId').agg(
        Total_Transaction_Amount=('Amount', 'sum'),
        Average_Transaction_Amount=('Amount', 'mean'),
        Transaction_Count=('Amount', 'count'),
        Std_Dev_Transaction_Amount=('Amount', 'std'),
        Max_Transaction_Amount=('Amount', 'max'),
        Min_Transaction_Amount=('Amount', 'min'),
        Median_Transaction_Amount=('Amount', 'median'),
        Transaction_Amount_Range=('Amount', lambda x: x.max() - x.min()),
        Skewness_Transaction_Amount=('Amount', 'skew'),
        Total_Unique_Products=('ProductId', 'nunique'),
        Total_Unique_Product_Categories=('ProductCategory', 'nunique')
    ).reset_index()

    # Fill NaN values in Standard Deviation and Skewness with 0 (for customers with a single transaction)
    aggregate_features['Std_Dev_Transaction_Amount'].fillna(0, inplace=True)
    aggregate_features['Skewness_Transaction_Amount'].fillna(0, inplace=True)

    print("Advanced aggregate features created successfully for each customer.")
    return aggregate_features
