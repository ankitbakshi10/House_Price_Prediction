import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def get_model_list():
    """
    Get the list of available models for house price prediction.
    
    Returns:
    --------
    model_list : list
        List of available model names
    """
    return [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost"
    ]

def train_model(X_train, y_train, X_test, y_test, model_type, hyperparams, cv=5):
    """
    Train a machine learning model for house price prediction.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_type : str
        Type of model to train (see get_model_list())
    hyperparams : dict
        Hyperparameters for the model
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    model : scikit-learn model
        The trained model
    metrics : dict
        Dictionary of evaluation metrics
    feature_importances : numpy.ndarray or None
        Feature importances if available, otherwise None
    """
    # Initialize model based on type
    if model_type == "Linear Regression":
        model = LinearRegression()
        
    elif model_type == "Ridge Regression":
        alpha = hyperparams.get('alpha', 1.0)
        model = Ridge(alpha=alpha)
        
    elif model_type == "Lasso Regression":
        alpha = hyperparams.get('alpha', 1.0)
        model = Lasso(alpha=alpha)
        
    elif model_type == "Decision Tree":
        max_depth = hyperparams.get('max_depth', 5)
        min_samples_split = hyperparams.get('min_samples_split', 2)
        
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        
    elif model_type == "Random Forest":
        n_estimators = hyperparams.get('n_estimators', 100)
        max_depth = hyperparams.get('max_depth', 5)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
    elif model_type == "Gradient Boosting":
        n_estimators = hyperparams.get('n_estimators', 100)
        learning_rate = hyperparams.get('learning_rate', 0.1)
        max_depth = hyperparams.get('max_depth', 3)
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        
    elif model_type == "XGBoost":
        n_estimators = hyperparams.get('n_estimators', 100)
        learning_rate = hyperparams.get('learning_rate', 0.1)
        max_depth = hyperparams.get('max_depth', 3)
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    # Get feature importances if available
    feature_importances = None
    
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
    elif model_type == "Linear Regression" or model_type == "Ridge Regression" or model_type == "Lasso Regression":
        # For linear models, use coefficients as feature importances
        feature_importances = np.abs(model.coef_)
    
    # Compile metrics
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, metrics, feature_importances
