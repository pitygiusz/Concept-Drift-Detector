import matplotlib.pyplot as plt
from datetime import datetime

from river import compose, feature_extraction, naive_bayes, utils, metrics, drift


def run_experiment(data_stream):
    
    model = compose.Pipeline(
        ('vectorizer', feature_extraction.TFIDF(lowercase=True)),
        ('classifier', naive_bayes.MultinomialNB())
    )
    
    metric = utils.Rolling(metrics.Accuracy(), window_size=50) 
    
    drift_detector = drift.ADWIN(delta=0.02) 
    
    dates = []
    accuracies = []
    drifts_dates = []
    
    latest_seen_date = datetime.fromisoformat(data_stream[0]['timestamp'].replace('Z', '+00:00'))
    
    for i, record in enumerate(data_stream):
        text = record['text']
        label = record['label'] 
        
        current_date = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        if current_date > latest_seen_date:
            latest_seen_date = current_date
            
        y_pred = model.predict_one(text)
        
        if y_pred is not None:
            metric.update(label, y_pred)
            
            error = 0 if y_pred == label else 1
            drift_detector.update(error)
            
            dates.append(latest_seen_date)
            accuracies.append(metric.get())
            
            if drift_detector.drift_detected:
                print(f"Detected Concept Drift on {latest_seen_date.strftime('%Y-%m-%d')}")
                drifts_dates.append(latest_seen_date)
        
        model.learn_one(text, label)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} articles.")

    return dates, accuracies, drifts_dates



def plot_results(dates, accuracies, drifts_dates, header):
    plt.figure(figsize=(14, 7))
    
    plt.plot(dates, accuracies, label='Rolling Accuracy (n=50)', color='blue', linewidth=2.5)
    
    tz = dates[0].tzinfo
    election_day = datetime(2024, 11, 5, tzinfo=tz)
    plt.axvline(x=election_day, color='purple', linestyle='-', linewidth=2, label='Election (05.11.2024)')

    inauguration_day = datetime(2025, 1, 20, tzinfo=tz)
    plt.axvline(x=inauguration_day, color='orange', linestyle='-', linewidth=2, label='Inauguration (20.01.2025)')

    for i, d in enumerate(drifts_dates):
        label = 'Concept Drift (ADWIN)' if i == 0 else ""
        plt.axvline(x=d, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label=label)


    plt.title(f'{header}', fontsize=16, fontweight='bold')
    plt.xlabel('Date of publication', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.0, 1.05)
    
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()