import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle



# Load the trained model
loaded_model = pickle.load(open('C:/Users/Vidisha/Desktop/Sustain&Grow/trained_model.sav', 'rb'))


# Load the training data to fit the imputer
df_train = pd.read_csv("C:/Users/Vidisha/Desktop//Sustain&Grow/esg.csv")

X_train = df_train.iloc[:, 1:11]  # Features: environmental, social, and governance factors

# Fit the SimpleImputer with training data
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

# Define a function to preprocess input data and make predictions
def predict_esg_score(input_data):
    # Prepare input data
    input_df = pd.DataFrame(input_data, index=[0])

    # Impute missing values
    input_imputed = imputer.transform(input_df)

    # Make prediction
    esg_score = loaded_model.predict(input_imputed)
    return esg_score[0]

# Example usage:
# Assume input_data is a dictionary containing user input, similar to the columns of the original dataframe
input_data = {
    'Emission': 5.6,
    'Innovation': 4.3,
    'Resource use': 6.7,
    'Human rights': 7.8,
    'Product responsibility': 6.5,
    'Workforce': 5.4,
    'Community': 7.2,
    'Management': 8.9,
    'Shareholders': 7.1,
    'CSR strategy': 6.2
}

# Predict ESG score
predicted_esg_score = predict_esg_score(input_data)
print("Predicted ESG Score:", predicted_esg_score)
