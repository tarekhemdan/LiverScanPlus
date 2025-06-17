# First install all required packages
!pip install -q torch torchvision torchaudio
!pip install -q monai albumentations opencv-python pandas matplotlib scikit-learn seaborn roboflow

import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from datetime import datetime
import seaborn as sns
from google.colab import files
import shutil
from roboflow import Roboflow
from tabulate import tabulate

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Enable benchmark for optimal performance

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

# Download dataset
data_path, data_yaml_path = download_dataset()

class LiverTumorDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None, img_size=256):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.img_size = img_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise ValueError(f"Failed to load image at {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Parse YOLO format label file
        label_path = self.label_paths[idx]
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h
                        
                        # Convert to coordinates
                        x1 = max(0, int(x_center - box_w/2))
                        y1 = max(0, int(y_center - box_h/2))
                        x2 = min(w, int(x_center + box_w/2))
                        y2 = min(h, int(y_center + box_h/2))
                        
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure proper tensor format
        mask = mask.float() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).float()
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        
        return image, mask

class DeepLabV3MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        mobilenet = mobilenet_v3_large(pretrained=True)
        self.backbone = mobilenet.features
        first_conv = self.backbone[0]
        self.backbone[0] = nn.Conv2d(3, first_conv.out_channels, 
                                   kernel_size=3, stride=2, padding=1, bias=False)
        nn.init.kaiming_normal_(self.backbone[0].weight)
        self.classifier = DeepLabHead(960, num_classes)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        out = self.classifier(x)
        return {'out': nn.functional.interpolate(out, size=input_shape, mode='bilinear')}

def calculate_map(preds, targets, iou_threshold=0.5):
    """Calculate mean Average Precision for segmentation masks"""
    ious = []
    for pred, target in zip(preds, targets):
        # Convert predictions to binary (0 or 1)
        pred_binary = (pred > 0.5).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)
    
    # Calculate precision and recall at different thresholds
    thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    for thresh in thresholds:
        tp = sum(iou >= thresh for iou in ious)
        fp = len(ious) - tp
        fn = len(targets) - tp
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        aps.append(precision)
    
    map50 = np.mean([iou >= iou_threshold for iou in ious])
    map50_95 = np.mean(aps)
    
    return map50, map50_95

class MonaiUNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='BATCH'
        )
        
    def forward(self, x):
        # MONAI UNet expects channel-first format
        if x.shape[1] != 3:  # If not already channel-first
            x = x.permute(0, 3, 1, 2)
        out = self.unet(x)
        return {'out': out}

def train_model(model, dataloaders, optimizer, num_epochs=50, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    best_iou = 0.0
    history = {'train': [], 'val': []}
    start_time = time.time()
    
    # Use Automatic Mixed Precision for faster training
    scaler = torch.amp.GradScaler()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, verbose=True)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            all_preds = []
            all_targets = []
            
            # Pre-allocate memory for metrics calculation
            tp_total, fp_total, fn_total = 0, 0, 0
            
            for inputs, masks in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True).float()
                masks = masks.to(device, non_blocking=True).float()
                
                optimizer.zero_grad(set_to_none=True)  # More efficient zero_grad
                
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Updated mixed precision
                        # Handle different model output formats
                        if hasattr(model, 'unet'):  # Check for MONAI UNet
                            inputs_perm = inputs.permute(0, 3, 1, 2) if inputs.shape[1] != 3 else inputs
                            outputs = model.unet(inputs_perm)
                            outputs = {'out': outputs}
                        else:
                            outputs = model(inputs)
                        
                        # Get the output tensor
                        out = outputs['out'] if isinstance(outputs, dict) else outputs
                        
                        # Prepare targets
                        if masks.dim() == 4:  # If masks have channel dimension
                            targets = masks.squeeze(1).long()
                        else:
                            targets = masks.long()
                        
                        loss = criterion(out, targets)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                
                # Get predictions and convert to numpy arrays
                preds = torch.sigmoid(out[:, 1]).detach()
                targets_np = targets.detach()
                
                # Calculate metrics in batches to reduce memory usage
                preds_binary = (preds > 0.5).float()
                targets_binary = (targets_np > 0.5).float()
                
                tp = torch.sum(preds_binary * targets_binary).item()
                fp = torch.sum(preds_binary * (1 - targets_binary)).item()
                fn = torch.sum((1 - preds_binary) * targets_binary).item()
                
                tp_total += tp
                fp_total += fp
                fn_total += fn
                
                # Store only necessary data for mAP calculation
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets_np.cpu().numpy())
            
            # Calculate metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            # Calculate mAP metrics
            map50, map50_95 = calculate_map(all_preds, all_targets)
            
            # Calculate precision, recall using accumulated values
            precision = tp_total / (tp_total + fp_total + 1e-6)
            recall = tp_total / (tp_total + fn_total + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            # Calculate speed (FPS)
            fps = len(dataloaders[phase].dataset) / (time.time() - epoch_start_time)
            
            # Memory usage
            memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            epoch_metrics = {
                'Mask_mAP@0.5': map50,
                'Mask_mAP@0.5:0.95': map50_95,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Speed_FPS': fps,
                'Memory_GB': memory,
                'loss': epoch_loss
            }
            
            print(f'{phase} Metrics:')
            for name, value in epoch_metrics.items():
                print(f'{name}: {value:.4f}', end=' | ')
            print()
            
            history[phase].append(epoch_metrics)
            
            if phase == 'val':
                scheduler.step(map50)  # Update learning rate based on validation mAP
                if map50 > best_iou:
                    best_iou = map50
                    torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
    
    train_time = time.time() - start_time
    return best_iou, history, train_time

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    eval_start = time.time()
    
    with torch.no_grad():
        for inputs, masks in dataloader:
            inputs = inputs.to(device).float()
            masks = masks.to(device).float()
            
            # Handle different model input formats
            if hasattr(model, 'unet'):  # MONAI UNet case
                # Permute to channel-first format
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = model.unet(inputs)
                outputs = {'out': outputs}
            else:
                outputs = model(inputs)
            
            # Get the output tensor
            out = outputs['out'] if isinstance(outputs, dict) else outputs
            
            # Get probability of class 1
            preds = torch.sigmoid(out[:, 1]).cpu().numpy()
            targets = masks.squeeze(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    # Convert to binary for metrics calculation
    preds_binary = [p > 0.5 for p in all_preds]
    targets_binary = [t > 0.5 for t in all_targets]
    
    # Calculate mAP metrics
    map50, map50_95 = calculate_map(all_preds, all_targets)
    
    # Calculate precision, recall using numpy logical operations
    tp = np.sum([np.logical_and(p, t).sum() for p, t in zip(preds_binary, targets_binary)])
    fp = np.sum([np.logical_and(p, np.logical_not(t)).sum() for p, t in zip(preds_binary, targets_binary)])
    fn = np.sum([np.logical_and(np.logical_not(p), t).sum() for p, t in zip(preds_binary, targets_binary)])
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Calculate speed (FPS)
    fps = len(dataloader.dataset) / (time.time() - eval_start)
    
    return {
        'Mask_mAP@0.5': map50,
        'Mask_mAP@0.5:0.95': map50_95,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Speed_FPS': fps,
        'Memory_GB': torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }

def save_results(results, history, model_name, train_time):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    
    # Add training time to results
    results['Train_Time'] = train_time
    
    # Save metrics to Excel
    df_results = pd.DataFrame(results, index=[0])
    df_results.to_excel(f'results/{model_name}_results_{timestamp}.xlsx')
    
    # Save training history to text file
    with open(f'results/{model_name}_history_{timestamp}.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training completed at: {timestamp}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n\n")
        
        for phase in history:
            f.write(f"{phase.upper()} HISTORY:\n")
            for epoch, metrics in enumerate(history[phase]):
                f.write(f"Epoch {epoch+1}:\n")
                for name, value in metrics.items():
                    f.write(f"{name}: {value:.4f}\n")
                f.write("\n")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    for metric in ['loss', 'Mask_mAP@0.5', 'F1-Score']:
        for phase in ['train', 'val']:
            values = [epoch[metric] for epoch in history[phase]]
            plt.plot(values, label=f'{phase} {metric}')
    plt.title(f'{model_name} Training Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f'results/{model_name}_training_curves_{timestamp}.png')
    plt.close()

def zip_and_download_results():
    # Create zip file
    shutil.make_archive('results', 'zip', 'results')
    
    # Download the zip file
    files.download('results.zip')
    
    # Clean up
    #os.remove('results.zip')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare datasets from the downloaded Roboflow data
    image_paths = []
    label_paths = []
    
    # Get all images and labels from train, valid, test folders
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(data_path, split, 'images')
        label_dir = os.path.join(data_path, split, 'labels')
        
        if os.path.exists(img_dir) and os.path.exists(label_dir):
            for img_file in os.listdir(img_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(img_file)[0]
                    label_file = base_name + '.txt'
                    
                    if os.path.exists(os.path.join(label_dir, label_file)):
                        image_paths.append(os.path.join(img_dir, img_file))
                        label_paths.append(os.path.join(label_dir, label_file))
    
    # Split data (we'll combine all splits and re-split for consistency)
    train_img, test_img, train_mask, test_mask = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42)
    train_img, val_img, train_mask, val_mask = train_test_split(
        train_img, train_mask, test_size=0.1, random_state=42)
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = LiverTumorDataset(train_img, train_mask, train_transform)
    val_dataset = LiverTumorDataset(val_img, val_mask, val_transform)
    test_dataset = LiverTumorDataset(test_img, test_mask, val_transform)
    
    # Use larger batch sizes if possible and pin memory for faster data transfer
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, 
                           num_workers=2, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=16, 
                         num_workers=2, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=16,
                         num_workers=2, pin_memory=True)
    }
    
    # Initialize models
    models = {
        'DeepLabV3_ResNet50': deeplabv3_resnet50(pretrained=True),
        'DeepLabV3_MobileNet': DeepLabV3MobileNet(num_classes=2),
        'nnUNet': UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='BATCH'
        )
    }
    
    # Modify classifier
    models['DeepLabV3_ResNet50'].classifier = DeepLabHead(2048, 2)
    
    # Train and evaluate
    all_results = {}
    for name, model in models.items():
        print(f'\n{"="*40}')
        print(f"Training {name}")
        print("="*40)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        best_iou, history, train_time = train_model(model, dataloaders, optimizer, device=device)
        
        model.load_state_dict(torch.load(f'best_{model.__class__.__name__}.pth'))
        test_metrics = evaluate_model(model, dataloaders['test'], device)
        
        # Add model parameters and training time
        test_metrics['Params_M'] = sum(p.numel() for p in model.parameters()) / 1e6
        test_metrics['Train_Time'] = train_time
        
        all_results[name] = test_metrics
        
        # Save individual model results
        save_results(test_metrics, history, name, train_time)
    
    # Save and display final results
    results_df = pd.DataFrame(all_results).T
    
    print("\nFinal Results:")
    print(results_df)
    
    # Save final comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_excel(f'results/final_comparison_{timestamp}.xlsx')
    results_df.to_csv(f'results/final_comparison_{timestamp}.txt', sep='\t')
    
    # Plot final comparison
    plt.figure(figsize=(12, 6))
    results_df[['Mask_mAP@0.5', 'Mask_mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/model_comparison_{timestamp}.png')
    plt.show()
    
    # Plot resource metrics
    plt.figure(figsize=(12, 6))
    results_df[['Speed_FPS', 'Params_M', 'Train_Time', 'Memory_GB']].plot(kind='bar')
    plt.title('Model Resource Usage Comparison')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/resource_comparison_{timestamp}.png')
    plt.show()
    
    # Zip and download results
    zip_and_download_results()