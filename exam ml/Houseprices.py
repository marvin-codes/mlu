# =========================================
# HOUSE PRICE PREDICTION PROJECT
# =========================================

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===============================
# PART A: DATA PREPARATION
# ===============================

print("Loading dataset...")
df = pd.read_csv("kc_house_data.csv")

# Display basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset summary statistics:")
print(df.describe())

print("\nDataset info:")
print(df.info())

# --------------------------------
# FEATURE DESCRIPTION (IMPORTANT)
# --------------------------------
print("\nFeature Descriptions:")
print("""
price: Target variable (house price)
bedrooms: Number of bedrooms
bathrooms: Number of bathrooms
sqft_living: Living area size
sqft_lot: Total lot size
floors: Number of floors
waterfront: Whether house has waterfront view
view: Quality of view
condition: Condition of the house
grade: Construction quality
sqft_above: Area above ground
sqft_basement: Basement area
yr_built: Year built
yr_renovated: Year renovated
zipcode: Location
lat, long: Geographical coordinates
""")

# ===============================
# DATA CLEANING
# ===============================

# Show dataset size BEFORE cleaning
print("\nBefore cleaning:", df.shape)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove duplicates
print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()

# Show dataset size AFTER cleaning
print("After cleaning:", df.shape)


# ===============================
# FEATURE SELECTION & SCALING
# ===============================

# Select numeric features only
numeric_df = df.select_dtypes(include=[np.number]).copy()

# Drop unnecessary column if present
if 'id' in numeric_df.columns:
    numeric_df = numeric_df.drop('id', axis=1)

# Define features and target
X = numeric_df.drop("price", axis=1)
y = numeric_df["price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling applied using StandardScaler.")
print("Scaling ensures all features are on the same scale and improves model performance.")


# ===============================
# TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTrain/Test sizes:")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ===============================
# PART B: EXPLORATORY DATA ANALYSIS
# ===============================

# Price distribution
plt.figure()
plt.hist(df["price"])
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

print("\nObservation: Price distribution is right-skewed (many low-mid prices, few very high prices).")

# Scatter plot 1
plt.figure()
plt.scatter(df["sqft_living"], df["price"])
plt.xlabel("sqft_living")
plt.ylabel("price")
plt.title("Living Area vs Price")
plt.show()

print("Observation: Strong positive linear relationship between living area and price.")

# Scatter plot 2
plt.figure()
plt.scatter(df["bedrooms"], df["price"])
plt.xlabel("bedrooms")
plt.ylabel("price")
plt.title("Bedrooms vs Price")
plt.show()

print("Observation: Weak to moderate relationship between bedrooms and price.")

# Correlation matrix
plt.figure(figsize=(10,6))
corr = numeric_df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("\nStrongest predictors of price:")
print(corr["price"].sort_values(ascending=False).head())

# Feature selection justification
print("\nFeatures like sqft_living, grade, and sqft_above have strong correlation with price.")

# Boxplot for outliers
plt.figure()
sns.boxplot(x=df["price"])
plt.title("Price Outliers")
plt.show()

print("Outliers detected. They may negatively affect regression models by skewing predictions.")

# Summary of findings
print("\nEDA Summary:")
print("""
- Larger houses tend to have higher prices.
- Strong correlation exists between size-related features and price.
- Outliers are present and may impact model performance.
""")


# ===============================
# PART C: MODEL TRAINING
# ===============================

# -------------------------------
# Linear Regression
# -------------------------------
start = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_time = time.time() - start

y_pred_lr = lr.predict(X_test)

# -------------------------------
# Decision Tree
# -------------------------------
start = time.time()
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
dt_time = time.time() - start

y_pred_dt = dt.predict(X_test)

print("\nTraining Time:")
print("Linear Regression:", lr_time)
print("Decision Tree:", dt_time)

print("\nModel Complexity:")
print("Linear Regression: Simple model")
print("Decision Tree: More complex and prone to overfitting")


# ===============================
# MODEL VISUALIZATION
# ===============================

# Linear Regression
plt.figure()
plt.scatter(y_test, y_pred_lr)
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# Decision Tree
plt.figure()
plt.scatter(y_test, y_pred_dt)
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

print("\nObservation:")
print("Linear Regression predictions are smoother.")
print("Decision Tree predictions show more variation.")


# ===============================
# RESULTS COMPARISON
# ===============================
results = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "LR_Pred": y_pred_lr,
    "DT_Pred": y_pred_dt
})

print("\nPrediction comparison:")
print(results.head())


# ===============================
# PART D: MODEL EVALUATION
# ===============================

# Linear Regression metrics
print("\nLinear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# Decision Tree metrics
print("\nDecision Tree Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_dt))
print("MAE:", mean_absolute_error(y_test, y_pred_dt))
print("R2 Score:", r2_score(y_test, y_pred_dt))


# ===============================
# RESIDUAL ANALYSIS
# ===============================
residuals = y_test - y_pred_lr

plt.figure()
plt.hist(residuals)
plt.title("Residual Distribution (Linear Regression)")
plt.show()

print("Residuals appear roughly normally distributed with some skew.")


# ===============================
# ERROR ANALYSIS
# ===============================

errors = abs(y_test - y_pred_lr)
print("\nLargest prediction errors:")
print(errors.sort_values(ascending=False).head())

print("""
Large errors may be caused by:
- Outliers
- Missing important features
- Non-linear relationships
""")

print("Model tends to underestimate high prices and overestimate low prices.")


# ===============================
# CROSS-VALIDATION
# ===============================
scores = cross_val_score(lr, X_scaled, y, cv=5)

print("\nCross-validation results:")
print("Mean Score:", scores.mean())
print("Standard Deviation:", scores.std())


# ===============================
# MODEL IMPROVEMENT (GRID SEARCH)
# ===============================
params = {'max_depth': [2, 4, 6, 8, 10]}

grid = GridSearchCV(
    DecisionTreeRegressor(),
    params,
    cv=5,
    scoring='neg_mean_squared_error'
)

grid.fit(X_train, y_train)

print("\nBest Decision Tree Depth:", grid.best_params_['max_depth'])

# Evaluate improved model
best_dt = grid.best_estimator_
y_pred_best = best_dt.predict(X_test)

print("\nImproved Decision Tree Performance:")
print("R2 Score:", r2_score(y_test, y_pred_best))


# ===============================
# FINAL CONCLUSION
# ===============================
print("""
Conclusion:
- Linear Regression provides stable and interpretable results.
- Decision Tree captures complex patterns but may overfit.
- Feature scaling and proper tuning improve model performance.
- Machine learning can effectively support house price prediction.
""")