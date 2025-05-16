from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import os

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('winequality.csv')

# Data preprocessing
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

df = df.drop('total sulfur dioxide', axis=1)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)

# Prepare features and target
features = df.drop(['quality', 'best quality'], axis=1)
features = features.fillna(features.mean())
target = df['best quality']

# Train model
imputer = SimpleImputer(strategy='mean')
norm = MinMaxScaler()
x = norm.fit_transform(imputer.fit_transform(features))
model = XGBClassifier()
model.fit(x, target)

# Save model and metadata
if not os.path.exists('models'):
    os.makedirs('models')
    
joblib.dump(model, 'models/wine_model.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
joblib.dump(norm, 'models/scaler.pkl')
joblib.dump(features.columns.tolist(), 'models/feature_names.pkl')  # Save feature names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['ph']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol']),
            'type': 1 if request.form['wine_type'] == 'white' else 0
        }
        
        # Load feature names and create DataFrame with correct order
        feature_names = joblib.load('models/feature_names.pkl')
        input_df = pd.DataFrame([form_data], columns=feature_names)
        
        # Load preprocessing objects and model
        imputer = joblib.load('models/imputer.pkl')
        scaler = joblib.load('models/scaler.pkl')
        model = joblib.load('models/wine_model.pkl')
        
        # Preprocess input
        processed_input = scaler.transform(imputer.transform(input_df))
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        
        # Prepare results
        result = {
            'prediction': 'High Quality' if prediction == 1 else 'Standard Quality',
            'probability': round(probability * 100, 2),
            'input_data': form_data
        }
        
        return render_template('results.html', result=result)
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)