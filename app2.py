from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    age = int(request.form['age'])
    gender = 1 if request.form['gender'] == 'Male' else 0
    subscription_length = int(request.form['subscription_length'])
    subscription_type = {'Basic': 0, 'Standard': 1, 'Premium': 2}[request.form['subscription_type']]
    number_of_logins = int(request.form['number_of_logins'])
    login_activity = 0 if request.form['login_activity'] == 'Active' else 1
    customer_ratings = float(request.form['customer_ratings'])

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'subscription_length': [subscription_length],
        'subscription_type': [subscription_type],
        'number_of_logins': [number_of_logins],
        'login_activity': [login_activity],
        'customer_ratings': [customer_ratings]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    churn_labels = ['No', 'Yes']
    
    # Fix the TypeError: Extract the scalar value from prediction
    churn_prediction = churn_labels[prediction[0]]
    churn_prob = prediction_proba[0][1] * 100
    no_churn_prob = prediction_proba[0][0] * 100

    return render_template('index.html', prediction_text=f"Prediction: {churn_prediction}",
                           churn_prob_text=f"Churn Probability: {churn_prob:.2f}%",
                           no_churn_prob_text=f"No Churn Probability: {no_churn_prob:.2f}%")

if __name__ == '__main__':
    app.run(debug=True)
