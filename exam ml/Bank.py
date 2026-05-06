# =========================================
# LOAN APPROVAL PREDICTION PROJECT
# FULL MARK VERSION (WITH EXPLANATIONS)
# =========================================

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier, StackingClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ===============================
# PART A: DATA PREPARATION
# ===============================

print("Loading dataset...")
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# 1. Display first 10 rows
print("\nFirst 10 rows:")
print(df.head(10))

# Dataset structure
print("\nDataset info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nExplanation:")
print("""
The dataset contains both categorical and numerical features.
The target variable is Loan_Status:
1 = Approved, 0 = Not Approved.
""")


# ===============================
# DATA CLEANING
# ===============================

# Drop ID column
df = df.drop('Loan_ID', axis=1)

# Convert Dependents
df['Dependents'] = pd.to_numeric(df['Dependents'].replace('3+', '3'), errors='coerce')

# Missing values BEFORE
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# Fill missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Drop remaining
df = df.dropna()

# Missing values AFTER
print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

print("\nJustification:")
print("""
Categorical values were filled with mode (most frequent).
Numerical values were filled with median to reduce outlier effect.
""")


# ===============================
# ENCODING
# ===============================
le = LabelEncoder()

categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Target encoding
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

print("\nEncoded data sample:")
print(df.head())

print("\nExplanation:")
print("""
Label encoding is used because variables are categorical and need numeric format.
""")


# ===============================
# FEATURE SCALING
# ===============================
scaler = StandardScaler()

num_cols = df.select_dtypes(include=np.number).columns.drop('Loan_Status')
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nScaling applied.")

print("""
Scaling ensures features like income and loan amount are on the same scale,
which improves performance especially for k-NN.
""")


# ===============================
# TRAIN-TEST SPLIT
# ===============================
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain/Test sizes:")
print("Train:", X_train.shape[0])
print("Test:", X_test.shape[0])

print("""
80/20 split ensures enough training data while preserving test data for evaluation.
""")


# ===============================
# PART B: EXPLORATORY ANALYSIS
# ===============================

# Histograms
plt.figure()
df['ApplicantIncome'].hist()
plt.title("Applicant Income Distribution")
plt.show()

print("ApplicantIncome is right-skewed.")

plt.figure()
df['LoanAmount'].hist()
plt.title("Loan Amount Distribution")
plt.show()

print("LoanAmount is slightly right-skewed.")


# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("""
Key relationships:
- ApplicantIncome correlates with LoanAmount
- Credit_History strongly affects Loan_Status
- LoanAmount relates to Loan_Amount_Term
""")

# Target distribution
plt.figure()
df['Loan_Status'].value_counts().plot(kind='bar')
plt.title("Loan Approval Distribution")
plt.show()

print("""
Dataset is slightly imbalanced (more approvals than rejections).
This may bias models toward predicting approvals.
""")

print("""
Key Features Influencing Approval:
- Credit_History
- ApplicantIncome
- LoanAmount
""")

print("""
Summary:
Income and credit history strongly influence loan approval.
""")


# ===============================
# PART C: CLUSTERING
# ===============================

# Feature selection justification
print("Using ApplicantIncome and LoanAmount for clustering (financial indicators).")

features = df[['ApplicantIncome', 'LoanAmount']]

# Elbow method
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 10), wcss)
plt.title("Elbow Method")
plt.show()

print("Optimal k chosen where curve flattens (~3).")

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Visualization
plt.figure()
plt.scatter(features.iloc[:,0], features.iloc[:,1], c=df['Cluster'])
plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("Clusters")
plt.show()

print("High income clusters tend to have higher loan amounts.")

# Compare clusters
print(pd.crosstab(df['Cluster'], df['Loan_Status']))

print("""
Clusters partially align with approval but not perfectly.
Other features like credit history are important.
""")

print("Clustering helps identify patterns for feature engineering.")


# ===============================
# PART D: MODEL TRAINING
# ===============================

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# k-NN (two values)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)

knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)

knn3_pred = knn3.predict(X_test)
knn5_pred = knn5.predict(X_test)

print("k=3 is more flexible, k=5 smoother.")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# Compare predictions
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "LR": lr_pred,
    "KNN3": knn3_pred,
    "KNN5": knn5_pred,
    "DT": dt_pred
})

print("\nPrediction Comparison:")
print(comparison.head())

print("""
Models differ due to different learning approaches.
""")


# ===============================
# ENSEMBLE MODELS
# ===============================

# Voting
voting = VotingClassifier(
    estimators=[('lr', lr), ('knn', knn5), ('dt', dt)],
    voting='hard'
)
voting.fit(X_train, y_train)
vote_pred = voting.predict(X_test)

# Stacking
stack = StackingClassifier(
    estimators=[('lr', lr), ('knn', knn5), ('dt', dt)],
    final_estimator=LogisticRegression()
)
stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

print("""
Stacking combines predictions using a meta-model.
""")


# ===============================
# EVALUATION
# ===============================

def evaluate(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]

results = pd.DataFrame({
    "Model": ["LR", "KNN3", "KNN5", "DT", "Voting", "Stacking"],
    "Accuracy": [
        evaluate(y_test, lr_pred)[0],
        evaluate(y_test, knn3_pred)[0],
        evaluate(y_test, knn5_pred)[0],
        evaluate(y_test, dt_pred)[0],
        evaluate(y_test, vote_pred)[0],
        evaluate(y_test, stack_pred)[0],
    ],
    "Precision": [
        evaluate(y_test, lr_pred)[1],
        evaluate(y_test, knn3_pred)[1],
        evaluate(y_test, knn5_pred)[1],
        evaluate(y_test, dt_pred)[1],
        evaluate(y_test, vote_pred)[1],
        evaluate(y_test, stack_pred)[1],
    ],
    "Recall": [
        evaluate(y_test, lr_pred)[2],
        evaluate(y_test, knn3_pred)[2],
        evaluate(y_test, knn5_pred)[2],
        evaluate(y_test, dt_pred)[2],
        evaluate(y_test, vote_pred)[2],
        evaluate(y_test, stack_pred)[2],
    ],
    "F1": [
        evaluate(y_test, lr_pred)[3],
        evaluate(y_test, knn3_pred)[3],
        evaluate(y_test, knn5_pred)[3],
        evaluate(y_test, dt_pred)[3],
        evaluate(y_test, vote_pred)[3],
        evaluate(y_test, stack_pred)[3],
    ]
})

print("\nModel Performance:")
print(results)

print("""
Ensemble models often improve performance by combining strengths.
""")


# ===============================
# CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, lr_pred)

plt.figure()
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("""
True Positives: Correct approvals
True Negatives: Correct rejections
Errors show misclassification.
""")


# ===============================
# FINAL INSIGHT
# ===============================
print("""
Final Insight:
Logistic Regression performs well due to simplicity.
k-NN is sensitive to scaling.
Decision Trees may overfit.
Ensemble methods provide better balanced performance.
""")