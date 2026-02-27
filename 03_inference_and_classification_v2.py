#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference and Classification Module V2
للاستخدام مع run_all.py
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

class YOLOInference:
    def __init__(self, model_path, confidence=0.5):
        """
        Initialize inference pipeline
        
        Args:
            model_path: path to trained YOLO model
            confidence: confidence threshold
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"✅ Model loaded: {model_path}")
    
    def predict_image(self, image_path):
        """
        Predict on single image
        
        Args:
            image_path: path to image
            
        Returns:
            results: YOLO results object
        """
        results = self.model.predict(
            source=image_path,
            conf=self.confidence,
            verbose=False
        )
        return results[0]
    
    def predict_batch(self, source_dir, output_dir=None):
        """
        Predict on batch of images
        
        Args:
            source_dir: directory containing images
            output_dir: directory to save results
            
        Returns:
            list of results
        """
        source_path = Path(source_dir)
        results_list = []
        
        # Get all images
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        
        print(f"\n{'='*60}")
        print(f"🔍 Running Inference on {len(image_files)} images")
        print(f"{'='*60}")
        
        for img_path in tqdm(image_files, desc="Processing"):
            result = self.predict_image(str(img_path))
            results_list.append({
                'image_name': img_path.name,
                'detections': len(result.boxes),
                'confidences': result.boxes.conf.tolist() if len(result.boxes) > 0 else []
            })
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            results = self.model.predict(
                source=str(source_path),
                conf=self.confidence,
                project=str(output_path),
                name="results",
                exist_ok=True,
                save=True,
                save_txt=True
            )
            print(f"✅ Results saved to {output_path / 'results'}")
        
        return results_list
    
    def create_submission(self, prediction_dir, output_file="submission.csv"):
        """
        Create submission CSV from predictions
        
        Args:
            prediction_dir: directory with prediction txt files
            output_file: output CSV filename
        """
        pred_path = Path(prediction_dir)
        submissions = []
        
        for txt_file in pred_path.glob("*.txt"):
            with open(txt_file) as f:
                predictions = f.read().strip()
            
            submissions.append({
                'image_id': txt_file.stem,
                'predictions': predictions if predictions else "no detections"
            })
        
        df = pd.DataFrame(submissions)
        df.to_csv(output_file, index=False)
        
        print(f"✅ Submission saved: {output_file}")
        print(f"   Total images: {len(df)}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference V2")
    parser.add_argument("--model", required=True, help="Path to trained YOLO model")
    parser.add_argument("--source", required=True, help="Source image or directory")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--submission", help="Output CSV file for submission")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = YOLOInference(args.model, confidence=args.confidence)
    
    # Run inference
    source_path = Path(args.source)
    
    if source_path.is_dir():
        results = inference.predict_batch(
            source_dir=args.source,
            output_dir=args.output
        )
        
        # Create submission
        if args.submission and args.output:
            pred_dir = Path(args.output) / "results" / "labels"
            inference.create_submission(pred_dir, args.submission)
    else:
        # Single image
        result = inference.predict_image(args.source)
        print(f"\n✅ Detections: {len(result.boxes)}")
        for box in result.boxes:
            print(f"   Class: {result.names[int(box.cls)]}, Confidence: {box.conf:.2f}")


if __name__ == "__main__":
    main()
