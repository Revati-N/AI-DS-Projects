# XGBoost AFTER FEATURE SELECTION
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

import pandas as pd
data = pd.read_csv('C:\DiabetesData\Diabetes_Dataset.csv')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

data = pd.get_dummies(data, columns=['Gender'])

X = data.drop('CLASS', axis=1)
y = data['CLASS']
# Remove leading/trailing spaces
y = y.str.strip()

# Convert categorical variable to numerical
y = y.map({'N': 0, 'P': 1, 'Y': 2})

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(importance_df)

# Select the top 4 features
top_4_features = importance_df['Feature'].iloc[:4]

# Select only the top 4 features from your train and test data
X_train_selected = X_train[top_4_features]
X_test_selected = X_test[top_4_features]

# Train and evaluate a new model on the selected features
model_selected = xgb.XGBClassifier()
model_selected.fit(X_train_selected, y_train)

y_pred = model_selected.predict(X_test_selected)

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy for XGBoost: ", accuracy)

# F1 Score
f1_score = metrics.f1_score(y_test, y_pred, average='weighted') # you can change the average parameter to 'micro', 'macro', 'weighted', depending on your problem
print("F1 Score for XGBoost: ", f1_score)

# Kappa Score
kappa_score = cohen_kappa_score(y_test, y_pred)
print("Kappa Score for XGBoost:", kappa_score)

# Precision
precision = metrics.precision_score(y_test, y_pred, average='weighted') # you can change the average parameter to 'micro', 'macro', 'weighted', depending on your problem
print("Precision for XGBoost: ", precision)

# Recall
recall = metrics.recall_score(y_test, y_pred, average='weighted') # you can change the average parameter to 'micro', 'macro', 'weighted', depending on your problem
print("Recall for XGBoost: ", recall)