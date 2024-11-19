import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xverse.transformer import WOE
from category_encoders.woe import WOEEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



def load_dataset(file_path):
    """Loads the dataset from the given file path and returns a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path and try again.")
    return pd.read_csv(file_path)

def create_advanced_aggregate_features(data):
    """
    Create aggregate features for customer transactions.
    """
    # Check for the existence of 'ProductId' column
    has_product_id = 'ProductId' in data.columns

    # Define aggregation dictionary
    agg_dict = {
        'Amount': ['sum', 'mean', 'count', 'std', 'max', 'min', 'median', 'skew'],
        'ProductCategory': ['nunique']
    }
    if has_product_id:
        agg_dict['ProductId'] = ['nunique']

    # Perform aggregation
    aggregate_features = data.groupby('CustomerId').agg(agg_dict)

    # Flatten column multi-index
    aggregate_features.columns = [
        f"{col[0]}_{col[1]}".replace('Amount_', '').replace('ProductCategory_', '')
        for col in aggregate_features.columns
    ]

    # Rename columns to more user-friendly names
    column_mapping = {
        'sum': 'Total_Transaction_Amount',
        'mean': 'Average_Transaction_Amount',
        'count': 'Transaction_Count',
        'std': 'Std_Dev_Transaction_Amount',
        'max': 'Max_Transaction_Amount',
        'min': 'Min_Transaction_Amount',
        'median': 'Median_Transaction_Amount',
        'skew': 'Skewness_Transaction_Amount',
        'nunique': 'Total_Unique_Products'
    }
    if has_product_id:
        column_mapping['ProductId_nunique'] = 'Total_Unique_Product_Ids'

    aggregate_features.rename(columns=column_mapping, inplace=True)

    # Reset index for compatibility
    aggregate_features.reset_index(inplace=True)

    return aggregate_features


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



def calculate_iv(df, feature, target):
    """
    Calculate the Information Value (IV) for a given feature and target.

    Args:
    - df (pd.DataFrame): The DataFrame containing the feature and target.
    - feature (str): The feature for which to calculate IV.
    - target (str): The target variable.

    Returns:
    - iv (float): The Information Value for the feature.
    """
    lst = []
    # Calculate the number of events and non-events
    total_events = df[target].sum()
    total_non_events = len(df[target]) - total_events
    
    for value in df[feature].unique():
        # Calculate events and non-events for each unique value in the feature
        events = df[(df[feature] == value) & (df[target] == 1)].shape[0]
        non_events = df[(df[feature] == value) & (df[target] == 0)].shape[0]
        
        # Calculate the proportion of events and non-events
        prop_events = events / total_events if total_events != 0 else 0
        prop_non_events = non_events / total_non_events if total_non_events != 0 else 0
        
        # Calculate WoE and IV
        woe = np.log((prop_events + 0.0001) / (prop_non_events + 0.0001))
        iv = (prop_events - prop_non_events) * woe
        
        # Store the results
        lst.append({'Value': value, 'Events': events, 'Non-Events': non_events,
                    'PropEvents': prop_events, 'PropNonEvents': prop_non_events,
                    'WoE': woe, 'IV': iv})
    
    # Create a DataFrame to hold the WoE and IV values for each unique value of the feature
    iv_df = pd.DataFrame(lst)
    return iv_df['IV'].sum()

def woe_binning(data, target):
    """
    Applies WoE binning and calculates Information Value (IV) for each feature.
    
    Args:
    - data (pd.DataFrame): The input DataFrame containing the features.
    - target (str): The target variable for WoE binning.
    
    Returns:
    - data_woe (pd.DataFrame): DataFrame with transformed features using WoE.
    - iv_df (pd.DataFrame): DataFrame containing IV values for each feature.
    """
    # Step 1: WoE Encoding
    woe_encoder = WOEEncoder(cols=data.columns.drop([target]))
    data_woe = woe_encoder.fit_transform(data.drop(target, axis=1), data[target])
    
    # Step 2: Calculate IV for Each Feature
    iv_values = []
    for feature in data.columns.drop([target]):
        iv = calculate_iv(data[[feature, target]], feature, target)
        iv_values.append(iv)
    
    # Create a DataFrame to store IV values
    iv_df = pd.DataFrame({
        'Feature': data.columns.drop([target]),
        'IV Value': iv_values
    })
    
    return data_woe, iv_df


def calculate_rfms(data, group_by_column):
    """
    Calculates RFMS score for each customer based on transaction data.
    Args:
    - data (pd.DataFrame): The input dataset containing customer transaction data.
    - group_by_column (str): Column name to group by (e.g., 'CustomerId').

    Returns:
    - rfms_df (pd.DataFrame): DataFrame with RFMS scores and default labels.
    """
    rfms_df = data.groupby(group_by_column).agg({
        'TransactionId': 'count',  # Frequency: Number of transactions
        'TransactionStartTime': 'max',  # Recency: Last transaction date
        'Amount': 'sum'  # Monetary: Total transaction amount
    }).reset_index()

    # Normalize the RFMS components
    rfms_df['Frequency_Score'] = rfms_df['TransactionId'] / rfms_df['TransactionId'].max()
    rfms_df['Monetary_Score'] = rfms_df['Amount'] / rfms_df['Amount'].max()
    rfms_df['Recency_Score'] = 1 - ((rfms_df['TransactionStartTime'].max() - rfms_df['TransactionStartTime']).dt.days / 
                                     (rfms_df['TransactionStartTime'].max() - rfms_df['TransactionStartTime'].min()).days)

    # Combine the RFMS scores into a single score
    rfms_df['RFMS_Score'] = 0.3 * rfms_df['Recency_Score'] + 0.4 * rfms_df['Frequency_Score'] + 0.3 * rfms_df['Monetary_Score']

    # Define a threshold for good and bad customers
    rfms_threshold = rfms_df['RFMS_Score'].median()
    rfms_df['Default_Indicator'] = rfms_df['RFMS_Score'].apply(lambda x: 'Good' if x >= rfms_threshold else 'Bad')

    return rfms_df

def normalize_features(data, columns):
    """
    Normalizes specified numerical columns using MinMaxScaler.
    Args:
    - data (pd.DataFrame): The dataset containing features to be normalized.
    - columns (list): List of columns to be normalized.

    Returns:
    - data (pd.DataFrame): Dataset with normalized columns.
    """
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def standardize_features(data, columns):
    """
    Standardizes specified numerical columns using StandardScaler.
    Args:
    - data (pd.DataFrame): The dataset containing features to be standardized.
    - columns (list): List of columns to be standardized.

    Returns:
    - data (pd.DataFrame): Dataset with standardized columns.
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data
