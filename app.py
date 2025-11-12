from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Automatically get training column names (saved during training)
try:
    dummy_columns = joblib.load('model_columns.pkl')
except:
    dummy_columns = model.feature_names_in_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect inputs
    math = float(request.form['math'])
    reading = float(request.form['reading'])
    writing = float(request.form['writing'])
    gender = request.form['gender']
    lunch = request.form['lunch']
    education = request.form['education']
    prep = request.form['prep']

    # Convert to DataFrame
    input_df = pd.DataFrame({
        'math score': [math],
        'reading score': [reading],
        'writing score': [writing],
        'gender': [gender],
        'lunch': [lunch],
        'test preparation course': [prep],
        'parental level of education': [education]
    })

    # One-hot encode + align columns
    input_encoded = pd.get_dummies(input_df)
    for col in dummy_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[dummy_columns]

    prediction = model.predict(input_encoded)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
