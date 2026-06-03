import matplotlib.pyplot as plt
from datetime import datetime


def plot_results(dates, accuracies, majority_accuracies, drifts_dates, header):
    plt.figure(figsize=(14, 7))
    
    plt.plot(dates, accuracies, label='Rolling Accuracy (n=200)', color='blue', linewidth=2)
    plt.plot(dates, majority_accuracies, label='Majority Class (n=200)', color='red', linewidth=2)
    
    tz = dates[0].tzinfo
    start_of_capaign_day = datetime(2024, 3, 5, tzinfo=tz)
    plt.axvline(x=start_of_capaign_day, color='brown', linestyle='-', linewidth=2, label='Start of Campaign', alpha=0.8)
    election_day = datetime(2024, 11, 5, tzinfo=tz)
    plt.axvline(x=election_day, color='purple', linestyle='-', linewidth=2, label='Election', alpha=0.8)
    inauguration_day = datetime(2025, 1, 20, tzinfo=tz)
    plt.axvline(x=inauguration_day, color='orange', linestyle='-', linewidth=2, label='Inauguration', alpha=0.8)
    hundred_days_day = datetime(2025, 4, 30, tzinfo=tz)
    plt.axvline(x=hundred_days_day, color='green', linestyle='-', linewidth=2, label='100 Days in Office', alpha=0.8)
    off_year_elections = datetime(2025, 11, 4, tzinfo=tz)
    plt.axvline(x=off_year_elections, color='skyblue', linestyle='-', linewidth=2, label='Off-Year Elections', alpha=0.8)


    for i, d in enumerate(drifts_dates):
        label = 'Concept Drift (ADWIN)' if i == 0 else ""
        plt.axvline(x=d, color='red', linestyle='--', alpha=1, linewidth=1.5, label=label)

    plt.title(f'{header}', fontsize=16, fontweight='bold')
    plt.xlabel('Date of publication', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.0, 1.05)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_shadow_models(steps, models_info, drifts_detected, drift_points):
    """Draws a multi-line chart comparing all generated models across multiple drifts."""
    print("\nGenerating the plot...")
    plt.figure(figsize=(14, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot lines for each model
    for idx, info in enumerate(models_info):
        color = colors[idx % len(colors)]
        label = f"Model {idx+1} (Active from step {info['start_step']})"
        plt.plot(steps, info['accuracies'], label=label, color=color, linewidth=2.5)

    # Vertical lines for ADWIN alerts
    for i, d in enumerate(drifts_detected):
        label = 'ADWIN Alert & Reset' if i == 0 else ""
        plt.axvline(x=d, color='red', linestyle='--', alpha=0.8, label=label)

    # Vertical lines for actual drift injections
    for i, dp in enumerate(drift_points):
        label = 'Actual Drift Injection' if i == 0 else ""
        plt.axvline(x=dp, color='purple', linestyle='-', linewidth=2, label=label)

    plt.title('Synthetic Stream + ADWIN Drift Detector', fontsize=16, fontweight='bold')
    plt.xlabel('Number of processed articles (Time)', fontsize=12)
    plt.ylabel('Rolling Accuracy (Window=100)', fontsize=12)
    plt.ylim(0.0, 1.05)
    
    # Place the legend outside if there are too many models
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()