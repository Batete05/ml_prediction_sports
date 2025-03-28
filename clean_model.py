import pandas as pd

# Load dataset
dataset = pd.read_csv('data/predictions.csv')

# Check for missing values
print("Missing values before processing:\n", dataset.isna().sum())

# Map categorical variables safely
if dataset['Sex'].dtype == 'object':
    dataset['Sex'] = dataset['Sex'].map({'Male': 1, 'Female': 0})

dataset['Prediction'] = dataset['Prediction'].map({'Basketball': 0, 'Football': 1, 'Hockey': 2})

# Check again for NaNs after mapping
print("Missing values after mapping:\n", dataset.isna().sum())

# Drop rows with NaNs
dataset.dropna(inplace=True)

# Verify dataset shape
print("Dataset shape after dropping NaNs:", dataset.shape)

# Save cleaned dataset
dataset.to_csv('data/cleaned_predictions.csv', index=False)
