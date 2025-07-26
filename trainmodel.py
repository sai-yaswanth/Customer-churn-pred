import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv('balanced_customer_churn.csv')

# Preprocessing
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])  # Male=1, Female=0
data['subscription_type'] = le.fit_transform(data['subscription_type'])  # Basic=0, Standard=1, Premium=2
data['login_activity'] = le.fit_transform(data['login_activity'])  # Low=0, Medium=1, High=2
data['churn'] = le.fit_transform(data['churn'])  # No=0, Yes=1

# Feature selection
X = data[['age', 'gender', 'subscription_length', 'subscription_type', 'number_of_logins', 'login_activity', 'customer_ratings']]
y = data['churn']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully!")
