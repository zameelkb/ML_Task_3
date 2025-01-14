# ML_Task_3

This project demonstrates a workflow for predicting California housing prices using machine learning. The code uses the California housing dataset, processes it, trains a Random Forest Regressor, evaluates its performance, and visualizes the results.

Features:-
Data Exploration: Inspect the first 25 rows and generate a statistical summary.
Preprocessing: Handles feature scaling and data splitting.
Modeling: Implements a Random Forest Regressor for prediction.
Evaluation: Calculates performance metrics like MAE, MSE, RMSE, and R².
Visualization: True vs. Predicted housing prices, Feature importance.
Libraries Used:-
Numpy: For numerical operations.
Matplotlib: For data visualization.
Seaborn: For enhanced visualizations.
Scikit-learn: For data preprocessing, modeling, and evaluation.

Dataset:- The California housing dataset is used,
fetched directly via sklearn.datasets.fetch_california_housing.

Step by step:-

1) Load the Dataset:

Fetches the California housing dataset.
Displays the first 25 rows and a statistical summary.
Checks for missing values.

2) Preprocessing:

Splits the data into features (X) and target (y).
Divides the data into training and testing sets.
Scales the features using StandardScaler.

3) Model Training:

Trains a RandomForestRegressor on the scaled training data.

4) Evaluation:

Predicts housing prices for the test set.
Evaluates the model using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)

5) Visualization:

Plots the true vs. predicted prices.
Displays the feature importance as a bar chart.
Results
The script outputs:

Performance metrics (MAE, MSE, RMSE, R²).
Visualizations:
Scatter plot comparing true and predicted housing prices.
Bar plot of feature importance.
