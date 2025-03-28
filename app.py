from flask import Flask, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset (or use your own dataset.csv)
data = pd.read_csv('data/predictions.csv')

def preprocess_data(data):
    # Convert 'Sex' into numerical values (Male = 1, Female = 0)
    data['Sex'] = data['Sex'].map({'Male': 1, 'Female': 0})
    X = data[['Age', 'Height', 'Sex']]  # Features
    y = data['Prediction']  # Target
    return X, y

def train_model():
    X, y = preprocess_data(data)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

@app.route('/')
def display_predictions():
    model = train_model()  # Train the model
    X, _ = preprocess_data(data)
    predictions = model.predict(X)
    
    # Add predictions to the data
    data['Prediction'] = predictions

    # Reset the index so 'Name' is a column again
    data.reset_index(inplace=True)

    # Convert data to a dictionary to pass to template
    predicted_sports = data[['Name', 'Age', 'Height', 'Sex', 'Prediction']].to_dict(orient='records')
    
    return render_template('predictions.html', predicted_sports=predicted_sports)

if __name__ == '__main__':
    app.run(debug=True)
