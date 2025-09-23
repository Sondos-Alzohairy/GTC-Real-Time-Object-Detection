import os 
import json
import requests
import zipfile
from pathlib import Path
import pandas as pd
from typing import Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class BDD100Dataset(Dataset):
    def __init__(self, data_dir, split="train", load_images=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.load_images = load_images

        # Target classes for autonomous driving
        self.class_names = ['person', 'car', 'truck', 'bus', 'traffic light', 'traffic sign']
        self.class2idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load annotations
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} images for {split}")

    def _load_annotations(self):
        ann_file = self.data_dir / "labels" / f"bdd100k_labels_images_{self.split}.json"
        
        with open(ann_file, "r") as f:
            annotations = json.load(f)
        
        # Keep only images with our target objects
        filtered = []
        for ann in annotations:
            for label in ann.get("labels", []):
                if label.get("category") in self.class_names and "box2d" in label:
                    filtered.append(ann)
                    break
        
        return filtered
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = self.data_dir / "images" / self.split / ann["name"]
           
        # Load image if needed
        if self.load_images:
            image_path = self.data_dir / "images" / "100k" / self.split / ann["name"]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None

        # Extract boxes and labels
        boxes = []
        labels = []
        for label in ann.get("labels", []):
            if label["category"] in self.class2idx and "box2d" in label:
                box = label["box2d"]
                boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
                labels.append(self.class2idx[label["category"]])

        return {
            "image": image,
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "image_id": ann["name"]
        }

def create_sample_dataset(data_dir: str, num_samples: int = 100):
        """
        Create a sample dataset for testing when BDD100K is not available.
        
        Args:
            data_dir: Directory to create sample dataset
            num_samples: Number of sample images to create
        """
        data_path = Path(data_dir)
        
        # Create directory structure
        (data_path / "images" / "100k" / "train").mkdir(parents=True, exist_ok=True)
        (data_path / "images" / "100k" / "val").mkdir(parents=True, exist_ok=True)
        (data_path / "labels").mkdir(parents=True, exist_ok=True)
        
        # Class names
        class_names = [
            'person', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
        ]
        
        # Create sample annotations and images
        for split in ['train', 'val']:
            split_samples = num_samples if split == 'train' else num_samples // 4
            annotations = []
            
            for i in range(split_samples):
                # Create sample image
                img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                img_name = f"sample_{i:04d}.jpg"
                img_path = data_path / "images" / "100k" / split / img_name
                
                cv2.imwrite(str(img_path), img)
                
                # Create sample annotations
                labels = []
                num_objects = np.random.randint(1, 6)  # 1-5 objects per image
                
                for j in range(num_objects):
                    # Random bounding box
                    x1 = np.random.randint(0, 1000)
                    y1 = np.random.randint(0, 500)
                    x2 = x1 + np.random.randint(50, 200)
                    y2 = y1 + np.random.randint(50, 150)
                    
                    # Ensure box is within image bounds
                    x2 = min(x2, 1280)
                    y2 = min(y2, 720)
                    
                    label = {
                        'category': np.random.choice(class_names),
                        'box2d': {
                            'x1': float(x1),
                            'y1': float(y1), 
                            'x2': float(x2),
                            'y2': float(y2)
                        }
                    }
                    labels.append(label)
                
                annotation = {
                    'name': img_name,
                    'labels': labels
                }
                annotations.append(annotation)
            
            # Save annotations
            ann_file = data_path / "labels" / f"bdd100k_labels_images_{split}.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f, indent=2)
        
        print(f"Created sample dataset with {num_samples} training and {num_samples//4} validation images")
        print(f"Dataset saved to: {data_path}")
def collate_fn(batch):
    """Simple collate function for DataLoader."""
    images = []
    targets = []
    
    for sample in batch:
        if sample['image'] is not None:
            image = torch.from_numpy(sample['image']).permute(2, 0, 1).float() / 255.0
            images.append(image)
        
        target = {
            'boxes': torch.from_numpy(sample['boxes']),
            'labels': torch.from_numpy(sample['labels']),
            'image_id': sample['image_id']
        }
        targets.append(target)
    
    return images, targets


def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """Create a DataLoader for the BDD100K dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

if __name__ == "__main__":
    # Simple test
    data_dir = "bdd100k_labels_release/bdd100k"
    
    # Load dataset
    dataset = BDD100Dataset(data_dir, split="train", load_images=False)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Classes: {dataset.class_names}")
    
    # Test first sample
    sample = dataset[0]
    print(f"Sample: {sample['image_id']}")
    print(f"Objects: {len(sample['labels'])}")