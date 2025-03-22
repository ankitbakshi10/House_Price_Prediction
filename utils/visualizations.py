import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation matrix heatmap for the given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to plot correlation matrix for
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    plt.figure(figsize=figsize)
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot the heatmap
    fig = plt.figure(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
                cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    return fig

def plot_feature_importance(importances, feature_names, figsize=(12, 8)):
    """
    Plot feature importances.
    
    Parameters:
    -----------
    importances : numpy.ndarray
        Array of feature importances
    feature_names : list or array-like
        Names of features
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    return fig

def plot_prediction_vs_actual(y_true, y_pred, figsize=(10, 6)):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.7)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Predicted vs Actual Values', fontsize=16)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def plot_residuals_vs_predicted(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    fig = plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(y_pred, residuals, alpha=0.7)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('Residuals vs Predicted Values', fontsize=16)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), figsize=(10, 6)):
    """
    Plot learning curve for a model.
    
    Parameters:
    -----------
    estimator : scikit-learn estimator
        The model to evaluate
    X : array-like
        Training data
    y : array-like
        Training target
    cv : int or cross-validation generator
        Cross-validation strategy
    train_sizes : array-like
        Points on the training set size to evaluate
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    from sklearn.model_selection import learning_curve
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='r2'
    )
    
    # Calculate mean and std for train and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Plot learning curve
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    # Plot std deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    plt.title('Learning Curve', fontsize=16)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig
