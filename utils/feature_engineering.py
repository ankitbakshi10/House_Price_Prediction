import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def create_features(features_df):
    """
    Create new features from existing ones. This could include interactions,
    polynomial features, or domain-specific transformations.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        The input features DataFrame
        
    Returns:
    --------
    enhanced_features : pandas.DataFrame
        DataFrame with additional engineered features
    """
    # Make a copy of the original features
    enhanced_features = features_df.copy()
    
    # Identify numerical columns for feature engineering
    numerical_cols = enhanced_features.select_dtypes(include=['int64', 'float64']).columns
    
    # Skip if there are not enough numerical columns
    if len(numerical_cols) < 2:
        return enhanced_features
    
    # Create interaction features for the most important numerical columns
    # (limit to avoid explosion of features)
    for i in range(min(len(numerical_cols), 3)):
        for j in range(i+1, min(len(numerical_cols), 4)):
            col_i = numerical_cols[i]
            col_j = numerical_cols[j]
            enhanced_features[f"{col_i}_{col_j}_interaction"] = enhanced_features[col_i] * enhanced_features[col_j]
    
    # Create polynomial features for the most important numerical columns
    for i in range(min(len(numerical_cols), 3)):
        col = numerical_cols[i]
        enhanced_features[f"{col}_squared"] = enhanced_features[col] ** 2
    
    # Create ratio features
    if len(numerical_cols) >= 2:
        col_1 = numerical_cols[0]
        col_2 = numerical_cols[1]
        # Avoid division by zero
        enhanced_features[f"{col_1}_to_{col_2}_ratio"] = enhanced_features[col_1] / enhanced_features[col_2].replace(0, 1)
    
    return enhanced_features

def feature_selection(features, target, method='None', k=5):
    """
    Perform feature selection to choose the most important features.
    
    Parameters:
    -----------
    features : pandas.DataFrame
        The input features DataFrame
    target : pandas.Series
        The target variable
    method : str
        Feature selection method (None, Select K Best, RFE, Feature Importance)
    k : int
        Number of features to select
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    selected_df : pandas.DataFrame
        DataFrame with only selected features
    """
    # Ensure k is not greater than the number of features
    k = min(k, features.shape[1])
    
    if method == 'None':
        # Return all features
        return features.columns.tolist(), features
    
    elif method == 'Select K Best':
        # Select K best features using f_regression
        selector = SelectKBest(f_regression, k=k)
        selector.fit(features, target)
        
        # Get mask of selected features
        selected_mask = selector.get_support()
        selected_features = features.columns[selected_mask].tolist()
        
    elif method == 'Recursive Feature Elimination':
        # Use RFE with Linear Regression
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=k)
        selector.fit(features, target)
        
        # Get mask of selected features
        selected_mask = selector.support_
        selected_features = features.columns[selected_mask].tolist()
        
    elif method == 'Feature Importance':
        # Use Random Forest feature importance
        model = RandomForestRegressor(random_state=42)
        model.fit(features, target)
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Select top k features
        selected_features = [features.columns[i] for i in indices[:k]]
    
    # Return selected features as list and DataFrame
    selected_df = features[selected_features]
    
    return selected_features, selected_df
