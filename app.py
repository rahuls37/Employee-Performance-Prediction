from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Load the model and encoders
model_path = "artifacts/model_trainer/linear_regression_model.joblib"
encoders_path = "artifacts/data_transformation/encoders.joblib"
prediction_pipeline = PredictionPipeline(model_path=model_path, encoders_path=encoders_path)

# Ensure the directory for storing previous predictions exists
os.makedirs("artifacts/predictions", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = prediction_pipeline.predict(input_df)

        # Save the input data and prediction
        input_df['Prediction'] = prediction
        input_df.to_csv('artifacts/predictions/predictions.csv', mode='a', header=not os.path.exists('artifacts/predictions/predictions.csv'), index=False)

        return render_template('result.html', prediction=prediction[0])

@app.route('/history')
def history():
    if os.path.exists('artifacts/predictions/predictions.csv'):
        history_df = pd.read_csv('artifacts/predictions/predictions.csv')
        return render_template('history.html', tables=[history_df.to_html(classes='data', header="true")])
    else:
        return "No history available."

if __name__ == '__main__':
    app.run(debug=True)