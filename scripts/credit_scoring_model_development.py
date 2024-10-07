import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(file_path):
    """Loads the dataset from the given file path and returns a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path and try again.")
    return pd.read_csv(file_path)


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
import pandas as pd

def extract_datetime_features(data, timestamp_column='TransactionStartTime'):

    # Ensure the timestamp column exists in the dataset
    if timestamp_column not in data.columns:
        raise ValueError(f"'{timestamp_column}' column not found in the dataset.")
    
    # Convert the timestamp column to datetime format if not already done
    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors='coerce')
    
    # Extract various datetime features
    data['Transaction_Hour'] = data[timestamp_column].dt.hour
    data['Transaction_Day'] = data[timestamp_column].dt.day
    data['Transaction_Month'] = data[timestamp_column].dt.month
    data['Transaction_Year'] = data[timestamp_column].dt.year
    data['Transaction_DayOfWeek'] = data[timestamp_column].dt.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
    data['Transaction_WeekOfYear'] = data[timestamp_column].dt.isocalendar().week  # Week of the year
    
    print("Datetime features extracted successfully.")
    
    return data


def encode_all_categorical(data, max_unique_values=10):

    # Identify all categorical columns in the dataset
    categorical_columns = data.select_dtypes(include='object').columns

    print(f"Found {len(categorical_columns)} categorical columns to encode: {list(categorical_columns)}")

    # Initialize Label Encoder
    label_encoder = LabelEncoder()
    
    # Iterate through each categorical column and apply the appropriate encoding
    for col in categorical_columns:
        unique_values = data[col].nunique()
        
        if unique_values <= max_unique_values:
            # Apply One-Hot Encoding if unique values are less than or equal to the threshold
            print(f"Applying One-Hot Encoding to '{col}' with {unique_values} unique values.")
            data = pd.get_dummies(data, columns=[col], drop_first=True)
        else:
            # Apply Label Encoding if unique values are greater than the threshold
            print(f"Applying Label Encoding to '{col}' with {unique_values} unique values.")
            data[col] = label_encoder.fit_transform(data[col].astype(str))  # Label Encoding
            print(f"Column '{col}' has been Label Encoded with {unique_values} unique values.")

    print("All categorical columns encoded successfully.")
    return data
def missing_values_summary(data):
  
    # Calculate the total and percentage of missing values for each column
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_values, '% of Total Values': missing_percent})
    
    # Filter to show only columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values(by='% of Total Values', ascending=False)
    
    print("Columns with Missing Values:")
    return missing_df




def scale_numerical_features(data, method='normalization', columns=None):
  
    # Select only numerical columns if not specified
    if columns is None:
        columns = data.select_dtypes(include='number').columns

    print(f"Scaling method: {method}")
    print(f"Columns to scale: {columns}")

    # Initialize the appropriate scaler
    if method == 'normalization':
        scaler = MinMaxScaler()
    elif method == 'standardization':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid method. Choose 'normalization' or 'standardization'.")

    # Fit and transform the data for the specified columns
    data[columns] = scaler.fit_transform(data[columns])

    print(f"Numerical columns scaled using {method}.")
    return data
