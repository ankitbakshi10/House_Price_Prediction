import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime

# Import utility modules
from utils.preprocessing import preprocess_data, handle_missing_values, detect_outliers_func, adjust_for_inflation
from utils.feature_engineering import create_features, feature_selection
from utils.model_training import train_model, get_model_list
from utils.visualizations import plot_correlation_matrix, plot_feature_importance, plot_prediction_vs_actual

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Application title and description
st.title("🏠 House Price Prediction")
st.markdown("""
This application helps you predict house prices using various machine learning algorithms. 
Upload your data, explore it, train models, and make predictions on new data.
""")

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = None
if 'feature_importances' not in st.session_state:
    st.session_state['feature_importances'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload & Preprocessing", "Exploratory Data Analysis", 
                                 "Model Training", "Prediction"])

# 1. DATA UPLOAD AND PREPROCESSING
if page == "Data Upload & Preprocessing":
    st.header("1. Data Upload & Preprocessing")
    
    # Data upload
    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    use_sample_data = st.checkbox("Use a sample dataset (Boston Housing)", value=False)
    
    if uploaded_file is not None:
        # Read the uploaded data
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.success("✅ Data successfully uploaded!")
    elif use_sample_data:
        # Using sklearn's Boston housing dataset as an example
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['PRICE'] = housing.target
        st.session_state['data'] = data
        st.success("✅ Sample California Housing dataset loaded!")
    
    # Display raw data if available
    if st.session_state['data'] is not None:
        data = st.session_state['data']
        
        st.subheader("Raw Data Preview")
        st.write(data.head())
        
        st.subheader("Data Information")
        buffer = StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Statistical Summary")
        st.write(data.describe())
        
        # Preprocessing options
        st.subheader("Data Preprocessing")
        
        # Target selection
        target_col = st.selectbox("Select the target column (house price):", data.columns.tolist())
        
        # Missing value handling
        st.markdown("#### Missing Values")
        missing_values = data.isnull().sum()
        
        if missing_values.sum() > 0:
            st.write("Missing values in each column:")
            st.write(missing_values[missing_values > 0])
            
            missing_strategy = st.selectbox(
                "Choose how to handle missing values:",
                ["Drop rows with missing values", "Fill numerical with mean", "Fill numerical with median"]
            )
        else:
            st.write("No missing values found in the dataset.")
            missing_strategy = "none"
        
        # Outlier detection
        st.markdown("#### Outlier Detection")
        detect_outliers_bool = st.checkbox("Detect outliers", value=True)
        outlier_treatment = st.selectbox(
            "Choose how to handle outliers:",
            ["None", "Remove outliers", "Cap outliers"]
        )
        
        # Feature scaling
        st.markdown("#### Feature Scaling")
        scaling_method = st.selectbox(
            "Choose scaling method for numerical features:",
            ["None", "StandardScaler", "MinMaxScaler"]
        )
        
        # Categorical encoding
        st.markdown("#### Categorical Features")
        categorical_encoding = st.selectbox(
            "Choose encoding method for categorical features:",
            ["One-Hot Encoding", "Label Encoding"]
        )
        
        # Preprocess button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Check if target is in the data
                if target_col not in data.columns:
                    st.error(f"Target column '{target_col}' not found in the data.")
                else:
                    # Perform preprocessing
                    features_df, target_series, preprocessed_df = preprocess_data(
                        data, 
                        target_col, 
                        missing_strategy,
                        detect_outliers=detect_outliers_bool,
                        outlier_treatment=outlier_treatment,
                        scaling_method=scaling_method,
                        categorical_encoding=categorical_encoding
                    )
                    
                    # Store in session state
                    st.session_state['preprocessed_data'] = preprocessed_df
                    st.session_state['features'] = features_df
                    st.session_state['target'] = target_series
                    
                    st.success("✅ Data preprocessing completed!")
                    
                    # Show preprocessed data
                    st.subheader("Preprocessed Data")
                    st.write(preprocessed_df.head())
                    
                    # Data shape after preprocessing
                    st.write(f"Original data shape: {data.shape}")

                    # Data shape after preprocessing
                    st.write(f"Preprocessed data shape: {preprocessed_df.shape}")

# 2. EXPLORATORY DATA ANALYSIS
elif page == "Exploratory Data Analysis":
    st.header("2. Exploratory Data Analysis")
    
    if st.session_state['preprocessed_data'] is None:
        st.warning("⚠️ Please upload and preprocess your data first.")
    else:
        preprocessed_data = st.session_state['preprocessed_data']
        features = st.session_state['features']
        target = st.session_state['target']
        
        # Display distribution of target variable
        st.subheader("Target Variable Distribution")
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(target, kde=True)
        plt.title(f"Distribution of Target Variable")
        st.pyplot(fig)
        
        # Statistical description of target
        st.subheader("Target Variable Statistics")
        st.write(target.describe())
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        # Heatmap of correlations
        st.write("Correlation Matrix Heatmap")
        corr_matrix = plot_correlation_matrix(preprocessed_data)
        st.pyplot(corr_matrix)
        
        # Top correlations with target
        st.write("Top Correlations with Target Variable")
        corr_with_target = preprocessed_data.corr()[target.name].sort_values(ascending=False)
        st.write(corr_with_target)
        
        # Feature histograms
        st.subheader("Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize:",
            options=features.columns.tolist(),
            default=features.columns.tolist()[:3]
        )
        
        if selected_features:
            feature_fig = plt.figure(figsize=(12, 4 * len(selected_features)))
            for i, feature in enumerate(selected_features):
                plt.subplot(len(selected_features), 1, i+1)
                sns.histplot(features[feature], kde=True)
                plt.title(f"Distribution of {feature}")
            plt.tight_layout()
            st.pyplot(feature_fig)
        
        # Scatter plots
        st.subheader("Relationship with Target")
        scatter_feature = st.selectbox(
            "Select a feature to plot against the target:",
            options=features.columns.tolist()
        )
        
        scatter_fig = plt.figure(figsize=(10, 6))
        plt.scatter(features[scatter_feature], target)
        plt.xlabel(scatter_feature)
        plt.ylabel(target.name)
        plt.title(f"{scatter_feature} vs {target.name}")
        st.pyplot(scatter_fig)
        
        # Interactive scatter plot with Plotly
        st.subheader("Interactive Feature Analysis")
        x_axis = st.selectbox("X-axis", features.columns.tolist())
        y_axis = st.selectbox("Y-axis", features.columns.tolist(), index=1 if len(features.columns) > 1 else 0)
        
        # Create a DataFrame combining features and target for plotting
        plot_df = pd.DataFrame(features)
        plot_df[target.name] = target.values
        
        fig = px.scatter(plot_df, x=x_axis, y=y_axis, color=target.name,
                         title=f"Scatter Plot: {x_axis} vs {y_axis}", 
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig)
        
        # Pair plot for selected features
        st.subheader("Pair Plot")
        pair_features = st.multiselect(
            "Select features for pair plot (select 2-4 features):",
            options=features.columns.tolist(),
            default=features.columns.tolist()[:3]
        )
        
        if len(pair_features) >= 2 and len(pair_features) <= 4:
            pair_df = features[pair_features].copy()
            pair_df[target.name] = target.values
            
            pair_fig = sns.pairplot(pair_df, hue=target.name)
            st.pyplot(pair_fig)
        elif len(pair_features) > 0:
            st.info("Please select between 2 and 4 features for the pair plot.")

# 3. MODEL TRAINING
elif page == "Model Training":
    st.header("3. Model Training")
    
    if st.session_state['features'] is None or st.session_state['target'] is None:
        st.warning("⚠️ Please upload and preprocess your data first.")
    else:
        features = st.session_state['features']
        target = st.session_state['target']
        
        # Feature selection
        st.subheader("Feature Selection")
        
        feature_selection_method = st.selectbox(
            "Choose a feature selection method:",
            ["None", "Select K Best", "Recursive Feature Elimination", "Feature Importance"]
        )
        
        if feature_selection_method != "None":
            k_features = st.slider(
                "Number of features to select:",
                min_value=1,
                max_value=len(features.columns),
                value=min(5, len(features.columns))
            )
            
            selected_features, selected_features_df = feature_selection(
                features, target, method=feature_selection_method, k=k_features
            )
            
            st.write("Selected Features:")
            st.write(selected_features)
            
            X = selected_features_df
        else:
            X = features
        
        # Train-test split
        st.subheader("Train-Test Split")
        
        test_size = st.slider("Test set size (%):", 10, 40, 20) / 100
        random_state = st.number_input("Random state (for reproducibility):", 0, 1000, 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=test_size, random_state=random_state
        )
        
        st.write(f"Training set size: {X_train.shape[0]} samples")
        st.write(f"Test set size: {X_test.shape[0]} samples")
        
        # Store split data in session state
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        
        # Model selection
        st.subheader("Model Selection")
        
        model_type = st.selectbox(
            "Choose a model type:",
            get_model_list()
        )
        
        # Configure hyperparameters
        st.subheader("Model Hyperparameters")
        
        hyperparams = {}
        
        if model_type == "Linear Regression":
            # No hyperparameters to tune
            pass
            
        elif model_type == "Ridge Regression":
            alpha = st.slider("Alpha:", 0.01, 10.0, 1.0, 0.01)
            hyperparams["alpha"] = alpha
            
        elif model_type == "Lasso Regression":
            alpha = st.slider("Alpha:", 0.01, 10.0, 1.0, 0.01)
            hyperparams["alpha"] = alpha
            
        elif model_type == "Decision Tree":
            max_depth = st.slider("Max depth:", 1, 20, 5)
            min_samples_split = st.slider("Min samples split:", 2, 20, 2)
            
            hyperparams["max_depth"] = max_depth
            hyperparams["min_samples_split"] = min_samples_split
            
        elif model_type == "Random Forest":
            n_estimators = st.slider("Number of trees:", 10, 200, 100)
            max_depth = st.slider("Max depth:", 1, 20, 5)
            
            hyperparams["n_estimators"] = n_estimators
            hyperparams["max_depth"] = max_depth
            
        elif model_type == "Gradient Boosting":
            n_estimators = st.slider("Number of estimators:", 10, 200, 100)
            learning_rate = st.slider("Learning rate:", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Max depth:", 1, 10, 3)
            
            hyperparams["n_estimators"] = n_estimators
            hyperparams["learning_rate"] = learning_rate
            hyperparams["max_depth"] = max_depth
            
        elif model_type == "XGBoost":
            n_estimators = st.slider("Number of estimators:", 10, 200, 100)
            learning_rate = st.slider("Learning rate:", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.slider("Max depth:", 1, 10, 3)
            
            hyperparams["n_estimators"] = n_estimators
            hyperparams["learning_rate"] = learning_rate
            hyperparams["max_depth"] = max_depth
        
        # Cross-validation
        st.subheader("Cross-Validation")
        cv_folds = st.slider("Number of CV folds:", 2, 10, 5)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train the model
                model, metrics, feature_importances = train_model(
                    X_train, y_train, X_test, y_test, 
                    model_type, hyperparams, cv_folds
                )
                
                # Store in session state
                st.session_state['model'] = model
                st.session_state['model_metrics'] = metrics
                st.session_state['feature_importances'] = feature_importances
                
                st.success("✅ Model training completed!")
        
        # Display metrics if model has been trained
        if st.session_state['model_metrics'] is not None:
            metrics = st.session_state['model_metrics']
            
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            with col2:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            
            with col3:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            
            # Cross-validation results
            st.subheader("Cross-Validation Results")
            st.write(f"Cross-Validation R² Scores: {metrics['cv_scores']}")
            st.write(f"Mean CV R² Score: {metrics['cv_mean']:.4f}")
            st.write(f"Standard Deviation: {metrics['cv_std']:.4f}")
            
            # Plot predicted vs actual values
            st.subheader("Predicted vs Actual Values")
            pred_vs_actual_fig = plot_prediction_vs_actual(
                metrics['y_test'], metrics['y_pred']
            )
            st.pyplot(pred_vs_actual_fig)
            
            # Feature importance plot
            if st.session_state['feature_importances'] is not None:
                st.subheader("Feature Importances")
                
                importance_fig = plot_feature_importance(
                    st.session_state['feature_importances'],
                    X_train.columns
                )
                
                st.pyplot(importance_fig)
            
            # Residual analysis
            st.subheader("Residual Analysis")
            
            # Calculate residuals
            residuals = metrics['y_test'] - metrics['y_pred']
            
            # Residual histogram
            hist_fig = plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.title("Residuals Distribution")
            plt.xlabel("Residual Value")
            st.pyplot(hist_fig)
            
            # Residual vs predicted
            scatter_fig = plt.figure(figsize=(10, 6))
            plt.scatter(metrics['y_pred'], residuals)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title("Residuals vs Predicted Values")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            st.pyplot(scatter_fig)

# 4. PREDICTION
elif page == "Prediction":
    st.header("4. Prediction")
    
    if st.session_state['model'] is None:
        st.warning("⚠️ Please train a model first.")
    else:
        model = st.session_state['model']
        X_train = st.session_state['X_train']
        features = X_train.columns.tolist()
        
        st.subheader("Enter House Details for Prediction")
        
        # Create input fields for each feature
        input_data = {}
        for feature in features:
            # Get min and max values from training data for sensible defaults
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            mean_val = float(X_train[feature].mean())
            
            # Create a slider for numerical inputs
            input_data[feature] = st.slider(
                f"{feature}:",
                min_val,
                max_val,
                mean_val,
                step=(max_val - min_val) / 100
            )
        
        # Add inflation adjustment options
        st.subheader("Inflation Adjustment Options")
        
        # Current year
        current_year = datetime.datetime.now().year
        
        # Base year (when the model is predicting for)
        base_year = st.number_input(
            "Base year of prediction (current dataset year):",
            min_value=1990,
            max_value=current_year,
            value=current_year,
            step=1
        )
        
        # Target year (future year to predict for)
        target_year = st.number_input(
            "Target year to adjust prediction for:",
            min_value=base_year,
            max_value=base_year + 50,
            value=base_year + 5,
            step=1
        )
        
        # Custom inflation rate
        inflation_rate = st.slider(
            "Average annual inflation rate (%):",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1
        ) / 100  # Convert percentage to decimal
        
        # Make prediction button
        if st.button("Predict Price"):
            # Create a dataframe from input
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Adjust for inflation if needed
            if base_year != target_year:
                inflation_adjusted_prediction = adjust_for_inflation(
                    prediction, 
                    base_year=base_year, 
                    target_year=target_year, 
                    avg_inflation_rate=inflation_rate
                )
            else:
                inflation_adjusted_prediction = prediction
            
            # Display the prediction
            st.subheader("Prediction Result")
            
            # Show original prediction
            st.success(f"Predicted house price (in {base_year}): ${prediction:,.2f}")
            
            # Show inflation-adjusted prediction if applicable
            if base_year != target_year:
                st.success(f"Inflation-adjusted price (in {target_year}): ${inflation_adjusted_prediction:,.2f}")
                
                # Show inflation rate and years
                st.info(f"Applied {inflation_rate*100:.1f}% annual inflation over {target_year - base_year} years.")
                
                # Calculate total percentage increase
                percentage_increase = ((inflation_adjusted_prediction / prediction) - 1) * 100
                st.info(f"Total price increase due to inflation: {percentage_increase:.1f}%")
            
            # Feature importance for this prediction
            if st.session_state['feature_importances'] is not None:
                st.subheader("Feature Contribution")
                
                # Create a DataFrame with feature names and importances
                importances = st.session_state['feature_importances']
                contributions = pd.DataFrame({
                    'Feature': features,
                    'Value': [input_data[f] for f in features],
                    'Importance': importances
                })
                
                contributions['Contribution'] = contributions['Value'] * contributions['Importance']
                contributions = contributions.sort_values('Contribution', ascending=False)
                
                # Display the contributions
                fig = px.bar(
                    contributions, 
                    x='Contribution', 
                    y='Feature',
                    orientation='h',
                    title='Feature Contributions to Prediction'
                )
                st.plotly_chart(fig)
        
        # What-if analysis
        st.subheader("What-If Analysis")
        st.write("Adjust the values below to see how they affect the prediction.")
        
        # Select features for what-if analysis
        what_if_features = st.multiselect(
            "Select features to analyze:",
            options=features,
            default=features[:3]
        )
        
        # Option to include inflation in what-if analysis
        include_inflation = st.checkbox("Include inflation adjustment in what-if analysis", value=True)
            
        if what_if_features and st.button("Run What-If Analysis"):
            what_if_fig = go.Figure()
            
            # Base prediction
            base_input = pd.DataFrame([input_data])
            base_prediction = model.predict(base_input)[0]
            
            # Apply inflation to base prediction if needed
            if include_inflation and base_year != target_year:
                base_adjusted_prediction = adjust_for_inflation(
                    base_prediction, 
                    base_year=base_year, 
                    target_year=target_year, 
                    avg_inflation_rate=inflation_rate
                )
            else:
                base_adjusted_prediction = base_prediction
            
            for feature in what_if_features:
                # Create range of values
                min_val = X_train[feature].min()
                max_val = X_train[feature].max()
                values = np.linspace(min_val, max_val, 20)
                
                predictions = []
                for val in values:
                    # Create a copy of the input with the changed feature
                    modified_input = base_input.copy()
                    modified_input[feature] = val
                    
                    # Make prediction
                    pred = model.predict(modified_input)[0]
                    
                    # Apply inflation adjustment if selected
                    if include_inflation and base_year != target_year:
                        pred = adjust_for_inflation(
                            pred, 
                            base_year=base_year, 
                            target_year=target_year, 
                            avg_inflation_rate=inflation_rate
                        )
                    
                    predictions.append(pred)
                
                # Add line to plot
                what_if_fig.add_trace(go.Scatter(
                    x=values,
                    y=predictions,
                    mode='lines',
                    name=feature
                ))
            
            # Set the chart title based on whether inflation is included
            if include_inflation and base_year != target_year:
                chart_title = f"What-If Analysis: How Feature Changes Affect Price (Adjusted to {target_year})"
            else:
                chart_title = "What-If Analysis: How Feature Changes Affect Price"
                
            what_if_fig.update_layout(
                title=chart_title,
                xaxis_title="Feature Value",
                yaxis_title="Predicted Price",
                legend_title="Feature"
            )
            
            # Add annotation for inflation info if applicable
            if include_inflation and base_year != target_year:
                what_if_fig.add_annotation(
                    x=0.5,
                    y=0.02,
                    xref="paper",
                    yref="paper",
                    text=f"Prices adjusted for inflation: {inflation_rate*100:.1f}% annually from {base_year} to {target_year}",
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            
            st.plotly_chart(what_if_fig)
