import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
df = pd.read_csv("/content/hr.csv")

# Define a function to calculate ESG scores
def calculate_esg_score(row):
    environmental_factors = row[1:4].astype(float)
    social_factors = row[4:8].astype(float)
    governance_factors = row[8:11].astype(float)

    environmental_score = np.nanmean(environmental_factors)
    social_score = np.nanmean(social_factors)
    governance_score = np.nanmean(governance_factors)

    esg_score = np.nanmean([environmental_score, social_score, governance_score])
    return esg_score

# Calculate ESG scores
df['ESG scores'] = df.apply(calculate_esg_score, axis=1)
print("ESG Scores:\n", df['ESG scores'])

# Drop rows with missing values in the target column
df.dropna(subset=['ESG scores'], inplace=True)

# Prepare the data
X = df.iloc[:, 1:11]  # Features: environmental, social, and governance factors
y = df['ESG scores']   # Target variable: ESG scores

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# Evaluate the model
train_score = model.score(X_train_imputed, y_train)
test_score = model.score(X_test_imputed, y_test)
print(f"Training R^2 score: {train_score}")
print(f"Testing R^2 score: {test_score}")

# Save the trained model using pickle
filename = "trained_model.sav"
pickle.dump(model, open(filename, 'wb'))
