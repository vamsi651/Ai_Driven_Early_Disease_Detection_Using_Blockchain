import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
import os

def train_models():
    if not os.path.exists('models'):
        os.makedirs('models')
        
    df = pd.read_csv('health_data.csv')
    
    # Features and Target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred)
    rf_rec = recall_score(y_test, rf_pred)
    
    joblib.dump(rf, 'models/rf_model.pkl')
    
    # SVM Classifier
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_prec = precision_score(y_test, svm_pred)
    svm_rec = recall_score(y_test, svm_pred)
    
    joblib.dump(svm, 'models/svm_model.pkl')
    
    metrics = {
        'Random Forest': {
            'Accuracy': round(rf_acc, 4),
            'Precision': round(rf_prec, 4),
            'Recall': round(rf_rec, 4)
        },
        'SVM': {
            'Accuracy': round(svm_acc, 4),
            'Precision': round(svm_prec, 4),
            'Recall': round(svm_rec, 4)
        }
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Models trained and saved.")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    train_models()
