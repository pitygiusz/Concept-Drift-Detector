from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

from river import metrics
from river import drift
from river import utils


def run_experiment(data_stream, model):
    

    
    metric = utils.Rolling(metrics.Accuracy(), window_size=200) 
    majority_metric = utils.Rolling(metrics.Accuracy(), window_size=200)
    
    drift_detector = drift.ADWIN(delta=0.002) 
    
    dates = []
    accuracies = []
    majority_accuracies = []
    drifts_dates = []
    
    # Track class counts for majority class prediction
    class_counts = {0: 0, 1: 0}
    
    latest_seen_date = datetime.fromisoformat(data_stream[0]['timestamp'].replace('Z', '+00:00'))
    
    for i, record in enumerate(data_stream):
        text = record['text']
        label = record['label'] 
        
        current_date = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        if current_date > latest_seen_date:
            latest_seen_date = current_date
            
        y_pred = model.predict_one(text)
        
        # Majority class prediction
        majority_class = max(class_counts, key=class_counts.get) if sum(class_counts.values()) > 0 else 0
        
        if y_pred is not None:
            metric.update(label, y_pred)
            majority_metric.update(label, majority_class)
            
            error = 0 if y_pred == label else 1
            drift_detector.update(error)
            
            dates.append(latest_seen_date)
            accuracies.append(metric.get())
            majority_accuracies.append(majority_metric.get())
            
            if drift_detector.drift_detected:
                print(f"Detected Concept Drift on {latest_seen_date.strftime('%Y-%m-%d')}")
                drifts_dates.append(latest_seen_date)
        
        model.learn_one(text, label)
        class_counts[label] += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} articles.")

    return dates, accuracies, majority_accuracies, drifts_dates