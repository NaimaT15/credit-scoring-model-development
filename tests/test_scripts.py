import pandas as pd
import pytest
from scripts.credit_scoring_model_development import create_advanced_aggregate_features

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'CustomerId': [1, 2, 3, 4, 5],
        'Amount': [100, 200, 300, 400, 500],
        'ProductCategory': ['A', 'B', 'C', 'A', 'B'],
        'TransactionStartTime': pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-01 11:00:00',
            '2023-01-01 12:00:00', '2023-01-01 13:00:00',
            '2023-01-01 14:00:00'
        ]),
        'ProductId': [101, 102, 103, 104, 105]  # Adding the missing column
    })

def test_create_advanced_aggregate_features(sample_data):
    """
    Test the advanced aggregate feature creation function.
    """
    features = create_advanced_aggregate_features(sample_data)

    # Check expected columns
    expected_columns = [
        'CustomerId',
        'Total_Transaction_Amount',
        'Average_Transaction_Amount',
        'Transaction_Count',
        'Std_Dev_Transaction_Amount',
        'Max_Transaction_Amount',
        'Min_Transaction_Amount',
        'Median_Transaction_Amount',
        'Skewness_Transaction_Amount',
        'Total_Unique_Products',
    ]
    if 'ProductId' in sample_data.columns:
        expected_columns.append('Total_Unique_Product_Ids')

    assert list(features.columns) == expected_columns

    # Validate some aggregated values
    assert features.loc[features['CustomerId'] == 1, 'Total_Transaction_Amount'].values[0] == 100
    assert features.loc[features['CustomerId'] == 5, 'Total_Unique_Product_Ids'].values[0] == 1
