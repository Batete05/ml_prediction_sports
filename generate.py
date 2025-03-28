import pandas as pd
import random
import os
from faker import Faker

# Initialize Faker instance
fake = Faker()

# Number of rows in the dataset
num_rows = 1000  # Adjust this for a larger dataset

# Ensure 'data/' directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Function to generate a random row
def generate_row():
    name = fake.name()  # Generate a random name
    age = random.randint(8, 70)  # Random age between 13 and 19
    height = random.randint(150, 200)  # Height between 150cm and 200cm
    sex = random.choice([0, 1])  # 0 for Female, 1 for Male
    prediction = random.choice(['Basketball', 'Football', 'Hockey'])  # Random sport prediction
    return [name, age, height, sex, prediction]

# Generate multiple rows
data = [generate_row() for _ in range(num_rows)]

# Create DataFrame
df = pd.DataFrame(data, columns=['Name', 'Age', 'Height', 'Sex', 'Prediction'])

# Save DataFrame to CSV in 'data/' folder
file_path = os.path.join(DATA_DIR, "predictions.csv")
df.to_csv(file_path, index=False)

print(f"Dataset created successfully! Saved to: {file_path}")
