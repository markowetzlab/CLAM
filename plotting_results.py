import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, brier_score_loss, balanced_accuracy_score, f1_score, roc_curve
import pickle
import os
import datetime

def load_splits_results(folder_path):
    split_files = [f for f in os.listdir(folder_path) if f.startswith("split_") and f.endswith("_results.pkl")]
    split_metrics = []
    roc_curves = []
    
    for split_file in split_files:
        split_path = os.path.join(folder_path, split_file)
        with open(split_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        metrics = calculate_metrics(data_dict)
        roc_curves.append(metrics.pop("ROC Curve"))  # Remove ROC Curve from table data
        split_metrics.append(metrics)
    
    return split_metrics, roc_curves

def calculate_metrics(data_dict):
    y_true = [entry['label'] for entry in data_dict.values()]
    y_prob = [entry['prob'][0][1] for entry in data_dict.values()]
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]
    
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Brier Score': brier_score_loss(y_true, y_prob),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics['ROC Curve'] = (fpr, tpr)
    
    return metrics

def summarize_metrics(split_metrics, results_folder):
    metrics_df = pd.DataFrame(split_metrics)
    
    avg_metrics = metrics_df.mean().to_frame().T
    avg_metrics.index = ['Average']
    
    std_metrics = metrics_df.std().to_frame().T
    std_metrics.index = ['Spread']
    
    final_metrics_df = pd.concat([avg_metrics, std_metrics, metrics_df], axis=0)
    
    csv_path = os.path.join(results_folder, "metrics_summary.csv")
    final_metrics_df.to_csv(csv_path)
    
    print(f"Metrics saved to {csv_path}")
    print(final_metrics_df.to_markdown())
    return final_metrics_df

def plot_roc_curves(roc_curves, results_folder):
    plt.figure(figsize=(8, 6))
    for fpr, tpr in roc_curves:
        plt.plot(fpr, tpr, alpha=0.3)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Splits')
    
    roc_all_path = os.path.join(results_folder, "roc_curves.png")
    plt.savefig(roc_all_path)
    plt.close()
    print(f"ROC Curves saved to {roc_all_path}")

def plot_avg_roc_curve(roc_curves, results_folder):
    avg_fpr = np.linspace(0, 1, 100)
    interp_tpr = []
    for fpr, tpr in roc_curves:
        interp_tpr.append(np.interp(avg_fpr, fpr, tpr))
    avg_tpr = np.mean(interp_tpr, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(avg_fpr, avg_tpr, label='Average ROC Curve', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve Across Splits')
    plt.legend()
    
    roc_avg_path = os.path.join(results_folder, "average_roc_curve.png")
    plt.savefig(roc_avg_path)
    plt.close()
    print(f"Average ROC Curve saved to {roc_avg_path}")

def evaluate_all_splits(folder_path):
    results_folder = os.path.join(os.path.dirname(folder_path), f"results_{datetime.datetime.now().strftime('%Y-%m-%d')}")
    os.makedirs(results_folder, exist_ok=True)
    
    split_metrics, roc_curves = load_splits_results(folder_path)
    metrics_df = summarize_metrics(split_metrics, results_folder)
    print("Plotting ROC Curves for All Splits")
    plot_roc_curves(roc_curves, results_folder)
    print("Plotting Average ROC Curve")
    plot_avg_roc_curve(roc_curves, results_folder)
    return metrics_df

evaluate_all_splits("/scratchc/fmlab/zuberi01/phd/CLAM/abmil_results/None_abmil")