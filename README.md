# 🏠House_Price_Prediction_Model
(Mini Project for college)
<br>
Project Link : [House_Price_Prediction](https://housepriceprediction-dgbqf8jrdnwcszfc7j4lz8.streamlit.app)
<br>
This project focuses on building a predictive model and deploying it as a web application that estimates real estate prices based on various property features using machine learning techniques. By analyzing historical housing data, the model helps users estimate property prices more accurately and make better investment decisions.

## 📌 Objectives

- Predict housing prices using historical data and multiple ML algorithms.
- Analyze feature importance to understand key price-driving factors.
- Compare different models to evaluate their performance.

  ## 🔍 Dataset

- **Source:** Kaggle - [House Price Dataset](https://www.kaggle.com/)
- **Features:** Location, number of rooms, area (sq ft), number of bedrooms, median salary, population, and more.
- **Target Variable:** Property Price

  ## 🧠 Machine Learning Models Used

- **Linear Regression** : Predicts target values by fitting a straight line through the data.
- **Lasso Regression** : A linear model that uses L1 regularization to reduce overfitting and perform feature selection.
- **Ridge Regression** : A linear model with L2 regularization that minimizes coefficients to handle multicollinearity.
- **Decision Tree Regressor** : Splits data into branches based on feature thresholds to predict continuous values.
- **Random Forest Regressor** : An ensemble of decision trees that improves accuracy and reduces overfitting by averaging predictions.
- **XGBoost Regressor** : A high-performance gradient boosting algorithm that builds trees sequentially to minimize prediction errors efficiently.

  ## 📈 Evaluation Metrics

- **R² Score** : Measures how well the model explains the variability of the target variable (higher is better).
- **Mean Absolute Error (MAE)** : The average of absolute differences between actual and predicted values.
- **Mean Squared Error (MSE)** : The average of squared differences between actual and predicted values (penalizes larger errors).
- **Root Mean Squared Error (RMSE)** : The square root of MSE, providing error in the same units as the target variable.

  ## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** 
  - `pandas`, `numpy` for data manipulation
  - `matplotlib`, `seaborn`, `plotly` for visualization
  - `scikit-learn`, `xgboost` for ML models
  - `streamlit` for web application
 
    ## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-estate-price-predictor.git
   cd real-estate-price-predictor
   
2. Install required packages:
   pip install -r requirements.txt
   
3. Run the model:
   python main.py

