from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import json
import os
from blockchain import Blockchain
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from database_utils import init_db, save_prediction

app = Flask(__name__)
blockchain = Blockchain()
init_db()

# Model Definitions
DISEASES = {
    'heart': {
        'name': 'Heart Disease',
        'features': ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'MaxHR'],
        'model_path': 'models/heart_model.pkl'
    },
    'kidney': {
        'name': 'Kidney Disease',
        'features': ['Age', 'BloodPressure', 'SpecificGravity', 'Albumin', 'Sugar', 'BloodGlucoseRandom'],
        'model_path': 'models/kidney_model.pkl'
    },
    'liver': {
        'name': 'Liver Disease',
        'features': ['Age', 'Gender', 'TotalBilirubin', 'DirectBilirubin', 'AlkalinePhosphotase', 'AlamineAminotransferase'],
        'model_path': 'models/liver_model.pkl'
    },
    'thyroid': {
        'name': 'Thyroid Disease',
        'features': ['Age', 'Sex', 'TSH', 'T3', 'TT4', 'FTI'],
        'model_path': 'models/thyroid_model.pkl'
    },
    'diabetes': {
        'name': 'Diabetes',
        'features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
        'model_path': 'models/diabetes_model.pkl'
    }
}

# Medically safe mid-range example placeholders
EXAMPLE_VALUES = {
    'Age': 'Eg: 45',
    'Sex': 'Eg: 1',
    'ChestPainType': 'Eg: 1',
    'RestingBP': 'Eg: 120 mmHg',
    'Cholesterol': 'Eg: 180 mg/dL',
    'MaxHR': 'Eg: 150',

    'BloodPressure': 'Eg: 72 mmHg',
    'SpecificGravity': 'Eg: 1.020',
    'Albumin': 'Eg: 1',
    'Sugar': 'Eg: 0',
    'BloodGlucoseRandom': 'Eg: 120 mg/dL',

    'Gender': 'Eg: 1',
    'TotalBilirubin': 'Eg: 0.8 mg/dL',
    'DirectBilirubin': 'Eg: 0.2 mg/dL',
    'AlkalinePhosphotase': 'Eg: 100 IU/L',
    'AlamineAminotransferase': 'Eg: 30 IU/L',

    'TSH': 'Eg: 2.5 mIU/L',
    'T3': 'Eg: 120 ng/dL',
    'TT4': 'Eg: 8 µg/dL',
    'FTI': 'Eg: 2.0',

    'Pregnancies': 'Eg: 2',
    'Glucose': 'Eg: 110 mg/dL',
    'SkinThickness': 'Eg: 20 mm',
    'Insulin': 'Eg: 85 µU/mL',
    'BMI': 'Eg: 23.5'
}

def load_disease_model(disease_key):
    try:
        model = joblib.load(DISEASES[disease_key]['model_path'])
        return model
    except Exception as e:
        print(f"Error loading {disease_key} model: {e}")
        return None

def get_metrics():
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def get_suggestions(disease_key, prediction, input_data):
    if prediction == 0:
        return ["Your results appear normal.", "Continue maintaining a healthy lifestyle and regular check-ups."]
    
    suggestions = ["Test result indicates potential risk. Please consult a specialist."]
    if disease_key == 'heart':
        if input_data.get('Cholesterol', 0) > 200: suggestions.append("High cholesterol detected; consider a low-fat diet.")
        if input_data.get('RestingBP', 0) > 130: suggestions.append("Elevated blood pressure; reduce salt intake.")
    elif disease_key == 'diabetes':
        if input_data.get('Glucose', 0) > 140: suggestions.append("Higher than normal glucose levels; monitor sugar intake.")
        if input_data.get('BMI', 0) > 25: suggestions.append("BMI is in the overweight range; consider regular exercise.")
    elif disease_key == 'kidney':
        if input_data.get('Albumin', 0) > 0: suggestions.append("Albumin detected in urine; further kidney tests recommended.")
    return suggestions

@app.route('/')
def index():
    return render_template('index.html', diseases=DISEASES)

@app.route('/predict/<disease_key>', methods=['GET', 'POST'])
def predict(disease_key):
    if disease_key not in DISEASES:
        return redirect(url_for('index'))
    
    disease_info = DISEASES[disease_key]
    
    if request.method == 'POST':
        try:
            input_values = []
            input_dict = {}
            for feat in disease_info['features']:
                val = float(request.form[feat])
                input_values.append(val)
                input_dict[feat] = val
                
            model = load_disease_model(disease_key)
            if not model:
                return "Model not found."
            
            prediction = int(model.predict([input_values])[0])
            prob = model.predict_proba([input_values])[0][1]
            
            result_text = "Positive" if prediction == 1 else "Negative"
            suggestions = get_suggestions(disease_key, prediction, input_dict)
            
            # Blockchain log
            log_entry = {
                'disease': disease_info['name'],
                'inputs': input_dict,
                'prediction': result_text,
                'probability': round(float(prob), 2)
            }
            blockchain.add_block(log_entry)
            # Permanent storage in SQLite
            hash_value = save_prediction(
                disease=disease_info['name'],
                confidence=round(float(prob), 2)
            )
            
            return render_template('results.html', 
                                 disease=disease_info['name'],
                                 prediction=result_text,
                                 probability=f"{prob*100:.1f}%",
                                 suggestions=suggestions,
                                 disease_key=disease_key)
        except Exception as e:
            return f"Error during prediction: {e}"
            
    return render_template('predict.html', disease=disease_info, example_values=EXAMPLE_VALUES)

@app.route('/dashboard')
def dashboard():
    chain_data = blockchain.get_chain_data()
    metrics = get_metrics()
    
    if metrics:
        names = list(metrics.keys())
        acc = [metrics[n]['Accuracy'] for n in names]
        f1 = [metrics[n]['F1'] for n in names]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, acc, width, label='Accuracy', color='#3498db')
        ax.bar(x + width/2, f1, width, label='F1 Score', color='#2ecc71')
        
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_ylim(0.9, 1.05)
        
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', transparent=True)
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        graph_url = None

    return render_template('dashboard.html', chain=chain_data, graph_url=graph_url, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
