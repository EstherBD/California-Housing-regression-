# California Housing (regression)
This project explores the **California Housing dataset** from `sklearn.datasets`, applying end-to-end machine learning to predict median house values. The notebook 
includes data cleaning, feature engineering (ratios, logs, interactions), nested CV (Linear, Ridge, RF, XGBoost), hyperparameter tuning, permutation importance, feature reduction, and final evaluation with Actual vs Predicted scatterplot.

## Steps in the Project

1. **Data Loading and Cleaning**
   - Loaded the California Housing dataset (`fetch_california_housing`).
   - Checked summary statistics, missing values, and distributions.
   - Renamed columns for readability.

2. **Exploratory Data Analysis**
   - Target distribution plot (Median House Value).
   - Correlation heatmap between all features and the target.
   - Quartile-based categorical split of Median Income.

3. **Feature Engineering**
   - **Log transforms**: `Population`, `AvgRooms`, `AvgBedrooms`, `MedianIncome`.
   - **Ratios**: `RoomsPerHousehold`, `BedroomsPerRoom`, `PopPerHousehold`.
   - **Density features**: `PeoplePerRoom`, `HouseholdDensity`.
   - **Interaction features**: `Income_x_Age`, `Lat_x_Lon`, `Income_x_Latitude`, `Income_x_Longitude`.
   - **Polynomial feature**: `IncomeSquared`.

4. **Modeling with Nested CV**
   - Models: **Linear Regression, Ridge, Random Forest, XGBoost**.
   - **Outer CV** for unbiased evaluation.
   - **Inner CV (GridSearchCV)** for hyperparameter tuning.
   - Collected coefficients (linear models) and feature importances (tree models).
   - Baseline (mean predictor) for comparison.

5. **Hyperparameter Tuning and Feature Importance**
   - Best hyperparameters selected per fold and overall.
   - Used **permutation importance** for feature selection (for the best model XGBoost).

6. **Feature Reduction and Final Model**
   - Selected top features: `Latitude`, `Longitude`, `Income_x_Longitude`, `Lat_x_Lon`, `AvgOccupancy`, `Income_x_Age`, `BedroomsPerRoom`, `Income_x_Latitude`, `RoomsPerHousehold`, `log_MedianIncome`, 'log_AvgRooms'.
   - Re-ran nested CV with **XGBoost on reduced features**.
   - Produced **Actual vs Predicted scatterplot**.

##  Results
- Compared performance across models (RMSE & R²).
- XGBoost with reduced features gave the best trade-off between accuracy and simplicity.
- Visualized predictions with **Actual vs Predicted plot**.

