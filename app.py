from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect scores from the form
    math = float(request.form['math'])
    reading = float(request.form['reading'])
    writing = float(request.form['writing'])

    # Predict race/ethnicity
    prediction = model.predict([[math, reading, writing]])[0]

    return render_template('index.html', prediction_text=f'Predicted Race/Ethnicity: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
