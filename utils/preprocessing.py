import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder

def preprocess_data(data, target_column, missing_strategy='none', 
                    detect_outliers=True, outlier_treatment='None',
                    scaling_method='None', categorical_encoding='One-Hot Encoding'):
    """
    Preprocess the input DataFrame by handling missing values, outliers,
    scaling numerical features, and encoding categorical features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input data to preprocess
    target_column : str
        The name of the target column
    missing_strategy : str
        Strategy to handle missing values (none, drop, mean, median)
    detect_outliers : bool
        Whether to detect outliers
    outlier_treatment : str
        Strategy to handle outliers (None, Remove outliers, Cap outliers)
    scaling_method : str
        Method to scale numerical features (None, StandardScaler, MinMaxScaler)
    categorical_encoding : str
        Method to encode categorical features (One-Hot Encoding, Label Encoding)
        
    Returns:
    --------
    features_df : pandas.DataFrame
        The preprocessed features
    target_series : pandas.Series
        The target variable
    preprocessed_df : pandas.DataFrame
        The complete preprocessed DataFrame (features + target)
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Separate target from features
    target_series = df[target_column]
    features_df = df.drop(columns=[target_column])
    
    # Handle missing values
    if missing_strategy != 'none':
        features_df, target_series = handle_missing_values(features_df, target_series, strategy=missing_strategy)
    
    # Detect and handle outliers if requested
    if detect_outliers and outlier_treatment != 'None':
        features_df, target_series = detect_outliers_func(features_df, target_series, treatment=outlier_treatment)
    
    # Identify numerical and categorical features
    numerical_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns
    
    # Scale numerical features if requested
    if scaling_method != 'None' and len(numerical_cols) > 0:
        if scaling_method == 'StandardScaler':
            scaler = StandardScaler()
            features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])
        elif scaling_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
            features_df[numerical_cols] = scaler.fit_transform(features_df[numerical_cols])
    
    # Encode categorical features if present
    if len(categorical_cols) > 0:
        if categorical_encoding == 'One-Hot Encoding':
            # Create dummy variables for categorical features
            features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
        elif categorical_encoding == 'Label Encoding':
            # Use label encoding for categorical features
            encoder = LabelEncoder()
            for col in categorical_cols:
                features_df[col] = encoder.fit_transform(features_df[col])
    
    # Create the complete preprocessed DataFrame
    preprocessed_df = features_df.copy()
    preprocessed_df[target_column] = target_series
    
    return features_df, target_series, preprocessed_df

def handle_missing_values(features, target, strategy='drop'):
    """
    Handle missing values in the dataset according to the specified strategy.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        Feature DataFrame
    target : pandas.Series
        Target variable
    strategy : str
        Strategy to handle missing values ('drop', 'mean', 'median')
        
    Returns:
    --------
    features : pandas.DataFrame
        Features with handled missing values
    target : pandas.Series
        Target with corresponding rows if needed
    """
    if strategy == 'Drop rows with missing values':
        # Identify rows with missing values
        missing_rows = features.isnull().any(axis=1)
        
        # Drop rows with missing values from both features and target
        features = features[~missing_rows].reset_index(drop=True)
        target = target[~missing_rows].reset_index(drop=True)
    
    elif strategy == 'Fill numerical with mean':
        # Fill missing values with column mean
        features = features.fillna(features.mean())
    
    elif strategy == 'Fill numerical with median':
        # Fill missing values with column median
        features = features.fillna(features.median())
    
    return features, target

def adjust_for_inflation(price, base_year=None, target_year=None, avg_inflation_rate=0.03):
    """
    Adjust a home price for inflation from base_year to target_year.
    
    Parameters:
    -----------
    price : float
        The original price to adjust
    base_year : int
        The year the original price is from (defaults to current year)
    target_year : int
        The year to adjust the price to (defaults to current year)
    avg_inflation_rate : float
        The average annual inflation rate (default: 3%)
        
    Returns:
    --------
    adjusted_price : float
        The inflation-adjusted price
    """
    # Default to current year if not specified
    current_year = datetime.datetime.now().year
    if base_year is None:
        base_year = current_year
    if target_year is None:
        target_year = current_year
    
    # If years are the same, no adjustment needed
    if base_year == target_year:
        return price
    
    # Calculate compound inflation over the years
    years_diff = target_year - base_year
    inflation_factor = (1 + avg_inflation_rate) ** years_diff
    
    # Adjust the price
    adjusted_price = price * inflation_factor
    
    return adjusted_price

def detect_outliers_func(features, target, treatment='None'):
    """
    Detect and handle outliers in the dataset using the IQR method.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        Feature DataFrame
    target : pandas.Series
        Target variable
    treatment : str
        Treatment method for outliers ('None', 'Remove outliers', 'Cap outliers')
        
    Returns:
    --------
    features : pandas.DataFrame
        Features with handled outliers
    target : pandas.Series
        Target with corresponding rows if needed
    """
    # Get numerical columns
    numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
    
    # Combine all rows that have outliers
    outlier_rows = pd.Series(False, index=features.index)
    
    for col in numerical_cols:
        # Calculate IQR for the column
        q1 = features[col].quantile(0.25)
        q3 = features[col].quantile(0.75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers
        col_outliers = (features[col] < lower_bound) | (features[col] > upper_bound)
        
        if treatment == 'Remove outliers':
            # Mark rows with outliers
            outlier_rows = outlier_rows | col_outliers
        
        elif treatment == 'Cap outliers':
            # Cap the outliers at the bounds
            features.loc[features[col] < lower_bound, col] = lower_bound
            features.loc[features[col] > upper_bound, col] = upper_bound
    
    if treatment == 'Remove outliers':
        # Remove outlier rows from both features and target
        features = features[~outlier_rows].reset_index(drop=True)
        target = target[~outlier_rows].reset_index(drop=True)
    
    return features, target
