import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

DATASET_PATH = 'data/sports_data.csv'
MODEL_PATH = 'ml_models/sport_pred_model.joblib'
SCALER_PATH = 'ml_models/scaler.pkl'

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Preprocess data (assuming columns are 'Age', 'Height', 'Sex', and 'Sport')
X = df[['Age', 'Height', 'Sex']]  # Features
y = df['Sport']  # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"Model and scaler saved successfully at {MODEL_PATH} and {SCALER_PATH}.")
