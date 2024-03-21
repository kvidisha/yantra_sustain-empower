import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Vidisha/Desktop/Sustain&Grow/trained_model.sav', 'rb'))

# Load the training data to fit the imputer
df_train = pd.read_csv("C:/Users/Vidisha/Desktop/Sustain&Grow/esg.csv")
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

def main():
    # Title
    st.title('ESG Score Predictor')
    st.write('This app predicts the ESG (Environmental, Social, and Governance) score based on input factors.')

    # Getting the input from the user
    st.sidebar.title('Input Factors')
    Emission = st.sidebar.text_input('Emission', value='5.6')
    Innovation = st.sidebar.text_input('Innovation', value='4.3')
    ResourceUse = st.sidebar.text_input('Resource Use', value='6.7')
    HumanRights = st.sidebar.text_input('Human rights', value='7.8')
    ProductResponsibility = st.sidebar.text_input('Product responsibility', value='6.5')
    Workforce = st.sidebar.text_input('Workforce', value='5.4')
    Community = st.sidebar.text_input('Community', value='7.2')
    Management = st.sidebar.text_input('Management', value='8.9')
    Shareholders = st.sidebar.text_input('Shareholders', value='7.1')
    CSRstrategy = st.sidebar.text_input('CSR strategy', value='6.2')    
    
    # Code for Prediction
    score = ''
    
    # Creating a button for Prediction
    if st.sidebar.button('Predict ESG Score'):
        input_data = {
            'Emission': float(Emission),
            'Innovation': float(Innovation),
            'Resource use': float(ResourceUse),
            'Human rights': float(HumanRights),
            'Product responsibility': float(ProductResponsibility),
            'Workforce': float(Workforce),
            'Community': float(Community),
            'Management': float(Management),
            'Shareholders': float(Shareholders),
            'CSR strategy': float(CSRstrategy)
        }
        
        score = predict_esg_score(input_data)
        
        st.success("Predicted ESG Score: {:.2f}".format(score))

if __name__ == '__main__':
    main()
