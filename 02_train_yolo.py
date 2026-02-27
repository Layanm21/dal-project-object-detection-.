#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Training Module
للاستخدام مع run_all.py
"""

from ultralytics import YOLO
from pathlib import Path
import argparse

def train_yolo(
    data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project_name="runs",
    model_name="yolov8m.pt"
):
    """
    Train YOLO model
    
    Args:
        data_yaml: path to data.yaml
        epochs: number of training epochs
        imgsz: image size
        batch: batch size
        device: device to use (0 for GPU)
        project_name: project directory name
        model_name: YOLO model size
    """
    print(f"\n{'='*60}")
    print("🎯 YOLO Training Module")
    print(f"{'='*60}")
    print(f"📊 Data: {data_yaml}")
    print(f"🔢 Epochs: {epochs}")
    print(f"📐 Image Size: {imgsz}")
    print(f"📦 Batch Size: {batch}")
    print(f"🖥️  Device: GPU {device}")
    print(f"🏗️  Model: {model_name}")
    
    # Load model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=5,
        device=device,
        project=project_name,
        name="train",
        exist_ok=True,
        verbose=True,
        save=True,
        save_txt=True,
        plots=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"📊 Results: {results}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model")
    
    args = parser.parse_args()
    
    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model
    )
