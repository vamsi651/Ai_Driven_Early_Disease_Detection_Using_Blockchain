import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json
import os

def train_model(disease_name, csv_file):
    print(f"Training model for {disease_name}...")
    df = pd.read_csv(csv_file)
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred)),
        'F1': float(f1_score(y_test, y_pred))
    }
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, f'models/{disease_name.lower()}_model.pkl')
    print(f"Model for {disease_name} saved. Accuracy: {metrics['Accuracy']:.4f}")
    return metrics

if __name__ == "__main__":
    all_metrics = {}
    diseases = {
        'Heart': 'heart_data.csv',
        'Kidney': 'kidney_data.csv',
        'Liver': 'liver_data.csv',
        'Thyroid': 'thyroid_data.csv',
        'Diabetes': 'diabetes_data.csv'
    }
    
    for name, file in diseases.items():
        all_metrics[name] = train_model(name, file)
        
    with open('models/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print("All models trained and metrics saved to models/metrics.json")
