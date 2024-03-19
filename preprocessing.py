import numpy as np
import pandas as pd
import os


def detect_features(df):
    """

    :param df: dataset with continuous and discrete features
    :return: return the column names with discrete and continuous data separately
    """
    continuous_features = []
    discrete_features = []

    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            continuous_features.append(col)
        elif df[col].nunique() < 10 and np.mean(np.isnan(df[col])) < 0.5:  # heuristic: if the number of unique values is less than 10 and the percentage of NaN values is less than 50%
            discrete_features.append(col)
        else:
            continue

    return continuous_features, discrete_features


def remove_null_rows(df):
    """

    :param df: Dataset to be checked for null values
    :return: return a dataset with records containing null values pruned out
    """
    # Detect null values in the DataFrame
    null_values = df.isnull().any(axis=1)

    # Remove rows containing null values
    df_cleaned = df[~null_values]

    return df_cleaned


def combine_datasets(real_data, synthetic_data, ratio, filename):
    """

    :param real_data: Original dataset
    :param synthetic_data: Dataset with synthetic data
    :param ratio: Percentage ratio of original data
    :return: return a new dataset comprising data points from the original and synthetic data
    """
    # Calculate the number of rows to take from each dataset
    n1 = int(len(real_data) * ratio)
    n2 = int(len(synthetic_data) * (1 - ratio))
    folder = 'sampled/mixed_tabular_samples'

    # Take the required number of rows from each dataset
    df1_sampled = real_data.sample(n=n1, random_state=1)
    df2_sampled = synthetic_data.sample(n=n2, random_state=1)

    # Combine the two datasets
    df_combined = pd.concat([df1_sampled, df2_sampled], ignore_index=True)

    # Create the new folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the file in the new folder
    filepath = os.path.join(folder, filename)
    df_combined.to_csv(filepath, index=False)

    return df_combined, filepath
