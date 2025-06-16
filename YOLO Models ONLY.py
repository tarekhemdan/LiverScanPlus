# Set the locale to UTF-8
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Check GPU
!nvidia-smi

# Clear output for better readability
from IPython.display import clear_output
clear_output()

# Install the necessary libraries
!pip install ultralytics roboflow opencv-python pandas matplotlib scikit-learn seaborn torchmetrics tabulate

"""# First, clean up the environment
!pip uninstall torch numpy -y
!pip install torch==2.0.1 numpy==1.23.5 --force-reinstall
!pip install ultralytics roboflow opencv-python pandas matplotlib scikit-learn seaborn torchmetrics tabulate

# GPU Setup and Verification
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Install required packages with version pinning
!pip install -q ultralytics roboflow opencv-python pandas matplotlib scikit-learn seaborn torchmetrics tabulate
!pip install -q torchvision==0.15.2
"""
# Import libraries
import torch
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import Image, display
from tabulate import tabulate
from roboflow import Roboflow
import time
from torchmetrics import JaccardIndex
from tabulate import tabulate

# Initialize random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##############################################################################
# Download Dataset (Liver Segmentation)
##############################################################################

def download_dataset():
    """Download liver segmentation dataset from Roboflow."""
    print("\n[1/4] Downloading dataset...")

    rf = Roboflow(api_key="xs542BAQQ4nGbfcWQdcq")
    project = rf.workspace("segmentasiliver").project("project-5c-liver-tumor-aikum")
    version = project.version(1)
    dataset = version.download("yolov8", overwrite=False)

    # Dataset stats
    splits = ['train', 'valid', 'test']
    stats = {}
    for split in splits:
        img_dir = os.path.join(dataset.location, split, 'images')
        if os.path.exists(img_dir):
            stats[split] = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))])

    print("\nDataset Statistics:")
    print(tabulate(pd.DataFrame([stats]), headers='keys', tablefmt='pretty', showindex=False))

    return dataset.location, os.path.join(dataset.location, 'data.yaml')

data_path, data_yaml_path = download_dataset()

##############################################################################
# Initialize YOLO Segmentation Models (v8, v9, v10)
##############################################################################

def initialize_yolo_models():
    """Initialize all YOLO segmentation models."""
    print("\n[2/4] Initializing YOLO models...")

    models = {
        'yolov8n-seg': YOLO('yolov8n-seg.pt'),
        'yolo11n-seg': YOLO('yolo11n-seg.pt'),
        'YOLOv12n-seg': YOLO('yolo12n-seg.yaml')
    }

    # Print model summaries
    print("\nModel Details:")
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.model.parameters())
        print(f"{name}: {param_count/1e6:.1f}M parameters")

    return models

models = initialize_yolo_models()

##############################################################################
# Train All YOLO Models
##############################################################################

def train_yolo_models(models):
    """Train all YOLO models with consistent settings."""
    print("\n[3/4] Training models...")

    # Common training parameters
    train_args = {
        'data': data_yaml_path,
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'workers': 2,
        'single_cls': True,
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'patience': 10,
        'augment': True
    }

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()

        # Train with progress bar
        results = model.train(
            **train_args,
            name=f'{name}_train',
            exist_ok=True
        )

        train_time = time.time() - start_time
        print(f"{name} training completed in {train_time/60:.1f} minutes")

        # Store training results
        models[name] = {
            'model': model,
            'train_time': train_time,
            'results': results
        }

    return models

models = train_yolo_models(models)

##############################################################################
# Evaluate All Models
##############################################################################

def evaluate_yolo_models(models):
    """Evaluate all YOLO models on test set."""
    print("\n[4/4] Evaluating models...")

    metrics = {}

    for name, config in models.items():
        print(f"\nEvaluating {name}...")
        start_time = time.time()

        # Evaluate with test-time augmentation
        results = config['model'].val(
            data=data_yaml_path,
            split='test',
            imgsz=640,
            batch=16,
            device=0 if torch.cuda.is_available() else 'cpu',
            rect=True
        )

        # Calculate FPS
        fps = 1000 / results.speed['inference']

        # Get segmentation metrics
        box_mp = results.box.mp  # mean precision
        box_mr = results.box.mr  # mean recall
        map50 = results.seg.map50  # mAP@0.5
        map50_95 = results.seg.map  # mAP@0.5:0.95

        metrics[name] = {
            'Mask_mAP@0.5': map50,
            'Mask_mAP@0.5:0.95': map50_95,
            'Precision': box_mp,
            'Recall': box_mr,
            'F1-Score': 2 * (box_mp * box_mr) / (box_mp + box_mr + 1e-16),
            'Speed_FPS': fps,
            'Params_M': sum(p.numel() for p in config['model'].model.parameters()) / 1e6,
            'Train_Time': config['train_time'],
            'Memory_GB': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }

        print(f"Test mAP@0.5: {map50:.3f} | mAP@0.5:0.95: {map50_95:.3f} | FPS: {fps:.1f}")

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Save metrics
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_df.to_csv(f'yolo_segmentation_comparison_{timestamp}.csv')

    print("\nModel Comparison Metrics:")
    print(tabulate(metrics_df, headers='keys', tablefmt='pretty'))

    return metrics_df

metrics_df = evaluate_yolo_models(models)

##############################################################################
# Enhanced Visualization
##############################################################################

def visualize_yolo_results(metrics_df):
    """Generate visualizations for YOLO comparison."""
    print("\nGenerating visualizations...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # 1. Performance Comparison
    plt.figure(figsize=(15, 10))

    metrics_to_plot = ['Mask_mAP@0.5', 'Speed_FPS', 'Params_M', 'Train_Time']
    titles = ['Mask mAP@0.5 (Higher is better)',
              'Inference Speed (FPS, Higher is better)',
              'Model Size (Parameters in Millions)',
              'Training Time (Seconds)']

    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles), 1):
        plt.subplot(2, 2, i)
        ax = sns.barplot(x=metrics_df.index, y=metrics_df[metric], palette='viridis')
        plt.title(title)
        plt.xticks(rotation=45)

        # Add value labels
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 5),
                       textcoords='offset points')

    plt.suptitle('YOLO Segmentation Model Comparison', y=1.02)
    plt.tight_layout()
    plt.savefig('yolo_segmentation_comparison.png', bbox_inches='tight')
    plt.show()

    # 2. Speed-Accuracy Tradeoff
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=metrics_df,
        x='Speed_FPS',
        y='Mask_mAP@0.5',
        hue=metrics_df.index,
        size='Params_M',
        sizes=(50, 300),
        palette='viridis'
    )

    plt.title('Speed vs Accuracy Tradeoff (Bubble size = Parameter count)')
    plt.grid(True, alpha=0.3)

    # Add labels
    for line in range(metrics_df.shape[0]):
        plt.text(
            metrics_df['Speed_FPS'][line]+0.5,
            metrics_df['Mask_mAP@0.5'][line],
            metrics_df.index[line],
            horizontalalignment='left',
            size='medium'
        )

    plt.tight_layout()
    plt.savefig('yolo_speed_vs_accuracy.png', bbox_inches='tight')
    plt.show()

visualize_yolo_results(metrics_df)

##############################################################################
# Export Results
##############################################################################

def export_results():
    """Package all results for download."""
    print("\nPackaging results...")

    # Create zip with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    !zip -r yolo_segmentation_results_{timestamp}.zip *.csv *.png

    # Download
    from google.colab import files
    files.download(f'yolo_segmentation_results_{timestamp}.zip')

export_results()

print("\n[COMPLETE] YOLO segmentation comparison finished!")