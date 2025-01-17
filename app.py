from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Database initialization function
def init_db():
    db_path = 'predictions.db'
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS predictions
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     input_data TEXT,
                     prediction REAL)''')
        conn.commit()
        conn.close()

# Initialize database when app starts
with app.app_context():
    init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and remove EmpNumber
        form_data = {k: [v] for k, v in request.form.items()}
        if 'EmpNumber' in form_data:
            del form_data['EmpNumber']
            
        # Create DataFrame
        input_df = pd.DataFrame(form_data)
        
        # Initialize prediction pipeline
        pipeline = PredictionPipeline(
            model_path='artifacts/model_trainer/linear_regression_model.joblib',
            encoders_path='artifacts/data_transformation/encoders.joblib'
        )
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        
        # Save to database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO predictions (timestamp, input_data, prediction) VALUES (?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(form_data), float(prediction))
        )
        conn.commit()
        conn.close()
        
        return render_template('result.html', prediction=prediction)
        
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/history')
def history():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    predictions = c.fetchall()
    conn.close()
    return render_template('history.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)