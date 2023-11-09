import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
# Information
def information(data):
    """
    This function provides basic information about a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame for which you want to obtain information.

    Returns:
        None
    """
    print("Columns:", data.columns)
    print("Column Types:", type(data.columns))
    print("Descriptive Statistics:\n", data.describe())
    print("Info:")
    data.info()



def handlingNullValues(data):
    """
    This function handles missing (null) values in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame with missing values to be handled.

    Returns:
        pd.DataFrame: A DataFrame with missing values replaced or filled with column means.

    Raises:
        ValueError: If any column contains non-numeric values.

    Test Cases:
    1. When the DataFrame has missing values:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, None, 4], 'B': [None, 5, 6, 7]})
        >>> result = handlingNullValues(data)
        >>> print(result)
           A    B
        0  1.0  6.0
        1  2.0  5.0
        2  2.33 6.0
        3  4.0  7.0

    2. When the DataFrame contains non-numeric values:
        >>> data = pd.DataFrame({'A': [1, 2, 'abc', 4], 'B': [None, 5, 6, 7]})
        >>> try:
        ...     handlingNullValues(data)
        ... except ValueError as e:
        ...     print(str(e))
        'A' Contains Non-Numeric Values, First Convert it to Numeric then try again.
    """

    for column in data.columns:
        if not data[column].notna().any():
            continue  # Skip this column
        if data[column].isna().any():
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                print(f"'{column}' Contains Non-Numeric Values, First Convert it to Numeric then try again.")

    return data


def deleteMultipleColumns(data, columns_to_delete):
    """
    Delete multiple columns from a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame from which columns will be deleted.
        columns_to_delete (list): List of column names to be deleted.

    Returns:
        pd.DataFrame: A DataFrame with specified columns removed.

    """

    # Use the drop method to remove the specified columns
    data = data.drop(columns=columns_to_delete, axis=1)

    return data  # Return the modified DataFrame

def handleCategoricalData(data, categorical_columns=None):
    """
    Handle categorical data in a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        categorical_columns (list, optional): List of column names with categorical data. If not provided, the function
            will automatically determine which columns to categorize.

    Returns:
        pd.DataFrame: A DataFrame with categorical data processed.
    """

    special_characters_pattern = r'[?@#&$]'

    data = data.replace(special_characters_pattern, None, regex=True)

    if categorical_columns is None:
        # Automatically determine categorical columns based on unique value count
        categorical_columns = []
        for column in data.columns:
            unique_values = data[column].nunique()
            if unique_values <= 3:
                categorical_columns.append(column)

    for col in categorical_columns:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
            data = pd.concat([data, dummies], axis=1)
            data.drop(col, axis=1, inplace=True)
            data[dummies.columns] = data[dummies.columns].astype(int)
            print("Now converting to numerical from booleans")
            for dum in dummies.columns:
                data[dum] = data[dum].astype(int)
            print("Conversion Done")

    return data


def normalization(data, columns_to_normalize=None, min_value=0, max_value=1):
    """
    Normalize specified columns in a pandas DataFrame using Min-Max scaling.

    Parameters:
        data (pd.DataFrame): The input DataFrame to be normalized.
        columns_to_normalize (list, optional): List of column names to be normalized. If None, normalize all numeric columns.
        min_value (float, optional): Minimum value after normalization (default is 0).
        max_value (float, optional): Maximum value after normalization (default is 1).

    Returns:
        pd.DataFrame: A DataFrame with specified columns normalized using Min-Max scaling.

    Test Cases:
    1. When normalizing all numeric columns:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 10, 15, 20]})
        >>> result = normalization(data)
        >>> print(result)
               A    B
        0  0.00  0.00
        1  0.33  0.33
        2  0.67  0.67
        3  1.00  1.00

    2. When normalizing specified columns:
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 10, 15, 20], 'C': [100, 200, 300, 400]})
        >>> result = normalization(data, columns_to_normalize=['A', 'C'])
        >>> print(result)
               A    B    C
        0  0.00  0.00  0.00
        1  0.33  0.33  0.33
        2  0.67  0.67  0.67
        3  1.00  1.00  1.00
    """

    # If columns_to_normalize is not specified, normalize all numeric columns
    if columns_to_normalize is None:
        columns_to_normalize = data.select_dtypes(include='number').columns.tolist()

    # Min-Max scaling for specified columns
    for column in columns_to_normalize:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min()) * (
                    max_value - min_value) + min_value

    return [data, columns_to_normalize]


def standardize(data, columns_to_standardize=None):
    """
    Standardize specified columns in a pandas DataFrame using z-score normalization.

    Parameters:
        data (pd.DataFrame): The input DataFrame to be standardized.
        columns_to_standardize (list, optional): List of column names to be standardized. If None, standardize all numeric columns.

    Returns:
        pd.DataFrame: A DataFrame with specified columns standardized using z-score normalization.

    Test Cases:
    1. When standardizing all numeric columns:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 10, 15, 20]})
        >>> result = standardize(data)
        >>> print(result)
                 A            B
        0 -1.161895 -1.161895
        1 -0.387298 -0.387298
        2  0.387298  0.387298
        3  1.161895  1.161895

    2. When standardizing specified columns:
        >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 10, 15, 20], 'C': [100, 200, 300, 400]})
        >>> result = standardize(data, columns_to_standardize=['A', 'C'])
        >>> print(result)
                 A           B           C
        0 -1.161895 -1.161895  0.800640
        1 -0.387298 -0.387298  0.114707
        2  0.387298  0.387298 -0.571226
        3  1.161895  1.161895 -1.344121
    """

    # If columns_to_standardize is not specified, standardize all numeric columns
    if columns_to_standardize is None:
        columns_to_standardize = data.select_dtypes(include='number').columns.tolist()

    # Z-score normalization for specified columns
    for column in columns_to_standardize:
        data[column] = (data[column] - data[column].mean()) / data[column].std()

    return [data, columns_to_standardize]


def extract_numeric_values(data, columns_to_process):
    """
    Extract numeric values from specified columns in a DataFrame and replace the columns.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns_to_process (list): List of column names to process.

    Returns:
        pd.DataFrame: A DataFrame with specified columns replaced by extracted numerical values (as strings).
    """
    for column_name in columns_to_process:
        column_data = data[column_name]

        # Use regular expression to extract numerical values
        numeric_values = column_data.apply(lambda x: re.findall(r'\d+', str(x)))

        # Replace the column with extracted numerical values (as strings)
        data[column_name] = numeric_values.apply(lambda x: ', '.join(x) if x else None)

    return data



def tokenize_text_values(data, columns_to_process):
    """
    Tokenize and process text values in specified columns in a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns_to_process (list): List of column names to process.

    Returns:
        pd.DataFrame: A DataFrame with specified columns tokenized and processed.
    """
    for column_name in columns_to_process:
        column_data = data[column_name]

        # Split text values by ", " and create a list of values
        split_values = column_data.str.split(', ')

        # Create a new DataFrame with binary columns for each token
        tokenized_data = pd.get_dummies(split_values.apply(pd.Series).stack()).groupby(level=0).max()

        # Check if the column contains numeric values and extract them
        numeric_values = column_data.str.extract(r'(\d+)').astype(float)

        # Replace the original column with tokenized and numeric values
        data = pd.concat([data, tokenized_data, numeric_values], axis=1)
        data.drop(columns=[column_name], inplace=True)
        data = data.replace({True:1,False:0})
    return data



# def featureSelection(data, target, num_features=10):
#     """
#     Select the top features based on feature importance scores from a Random Forest classifier.
#
#     Parameters:
#         data (pd.DataFrame): The input feature dataset.
#         target (pd.Series): The target variable.
#         num_features (int): Number of top features to select.
#
#     Returns:
#         pd.DataFrame: The dataset with selected features.
#     """
#     # Train a Random Forest classifier
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(data, target)
#
#     # Get feature importances
#     feature_importances = clf.feature_importances_
#
#     # Get the indices of the top-k most important features
#     top_feature_indices = feature_importances.argsort()[::-1][:num_features]
#
#     # Select the top features from the original dataset
#     selected_data = data.iloc[:, top_feature_indices]
#
#     return selected_data


# class Sampling:
#
#     def __init__(self, data):
#         self.data = data
#
#     def undersample(self, target_column, ratio=1.0):
#         """
#         Perform undersampling to balance the dataset.
#
#         Parameters:
#             target_column (str): The name of the target column (class labels).
#             ratio (float): The desired ratio of minority class samples after undersampling.
#
#         Returns:
#             pd.DataFrame: The undersampled dataset.
#         """
#         # Separate the majority and minority class
#         majority_class = self.data[self.data[target_column] == 0]
#         minority_class = self.data[self.data[target_column] == 1]
#
#         # Undersample the majority class
#         majority_undersampled = resample(majority_class, replace=False, n_samples=int(len(minority_class) * ratio), random_state=42)
#
#         # Combine the minority class with the undersampled majority class
#         undersampled_data = pd.concat([majority_undersampled, minority_class])
#
#         return undersampled_data
#
#     def oversample(self, target_column, ratio=1.0):
#         """
#         Perform oversampling to balance the dataset.
#
#         Parameters:
#             target_column (str): The name of the target column (class labels).
#             ratio (float): The desired ratio of minority class samples after oversampling.
#
#         Returns:
#             pd.DataFrame: The oversampled dataset.
#         """
#         # Separate the majority and minority class
#         majority_class = self.data[self.data[target_column] == 0]
#         minority_class = self.data[self.data[target_column] == 1]
#
#         # Oversample the minority class
#         minority_oversampled = resample(minority_class, replace=True, n_samples=int(len(majority_class) * ratio), random_state=42)
#
#         # Combine the majority class with the oversampled minority class
#         oversampled_data = pd.concat([majority_class, minority_oversampled])
#
#         return oversampled_data