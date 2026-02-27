#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🚀 DAL Shemagh Detection - السترة التقليدية السعودية
Main script to download, train, and run inference on YOLO model
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil

# Avoid UnicodeEncodeError on terminals using legacy encodings (e.g., cp1256).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

class DALProject:
    def __init__(self, project_dir=None):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.data_dir = self.project_dir / "data"
        self.images_dir = self.project_dir / "images"
        self.labels_dir = self.project_dir / "labels"
        self.models_dir = self.project_dir / "models"
        self.competition_name = "dal-shemagh-detection-challenge"
        
        # Create directories
        for d in [self.data_dir, self.models_dir]:
            d.mkdir(exist_ok=True)

    def setup_kaggle_api(self):
        """Verify Kaggle API is configured"""
        print("\n" + "="*60)
        print("🔐 Checking Kaggle API Setup...")
        print("="*60)
        
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        if not kaggle_json.exists():
            print("❌ kaggle.json not found!")
            print(f"   Expected location: {kaggle_json}")
            print("\n   📝 Create it by:")
            print("   1. Go to: https://www.kaggle.com/settings")
            print("   2. Click 'Create New API Token'")
            print("   3. Place kaggle.json in: ~/.kaggle/")
            return False
        
        os.chmod(kaggle_json, 0o600)
        print("✅ Kaggle API configured!")
        return True

    def download_data(self):
        """Download dataset from Kaggle"""
        print("\n" + "="*60)
        print("📥 Downloading Dataset from Kaggle...")
        print("="*60)
        print(f"   Competition: {self.competition_name}")
        
        try:
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c", self.competition_name,
                "-p", str(self.data_dir),
                "--force"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Download failed!")
                print(f"   Error: {result.stderr}")
                return False
            
            print("✅ Dataset downloaded successfully!")
            
            self.extract_downloaded_archives()
            self.prepare_dataset_layout()
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def extract_downloaded_archives(self):
        """Extract zip archives inside data directory recursively"""
        import zipfile

        zip_files = list(self.data_dir.rglob("*.zip"))
        if not zip_files:
            print("   ℹ️  No zip files found to extract")
            return

        for zip_file in zip_files:
            print(f"   📦 Extracting {zip_file.relative_to(self.project_dir)}...")
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(zip_file.parent)
            zip_file.unlink()

    def prepare_dataset_layout(self):
        """
        Make sure images/ and labels/ exist directly under project_dir.
        Kaggle downloads are usually extracted under data/, so we move them once.
        """
        print("\n" + "=" * 60)
        print("🧭 Preparing Dataset Layout...")
        print("=" * 60)

        for folder_name in ["images", "labels"]:
            target = self.project_dir / folder_name
            if target.exists():
                print(f"   ✅ {folder_name}/ already exists at project root")
                continue

            candidates = [p for p in self.data_dir.rglob(folder_name) if p.is_dir()]
            if not candidates:
                print(f"   ⚠️  Could not find {folder_name}/ under data/")
                continue

            # Prefer the most dataset-like candidate (contains split folders)
            candidates.sort(
                key=lambda p: sum((p / split).exists() for split in ["train", "val", "test"]),
                reverse=True,
            )
            source = candidates[0]
            print(f"   📁 Moving {source.relative_to(self.project_dir)} -> {target.relative_to(self.project_dir)}")
            shutil.move(str(source), str(target))

    def install_requirements(self):
        """Install required packages"""
        print("\n" + "="*60)
        print("📦 Installing Requirements...")
        print("="*60)
        
        req_file = self.project_dir / "requirements.txt"
        if not req_file.exists():
            print(f"❌ requirements.txt not found at {req_file}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)]
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                print("✅ Requirements installed!")
                return True
            else:
                print("❌ Installation failed!")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        print("\n" + "="*60)
        print("⚙️  Creating data.yaml...")
        print("="*60)
        
        data_yaml_path = self.project_dir / "data.yaml"

        default_names = ["shemagh", "ghutra", "dal", "traditional-headdress"]
        nc = len(default_names)
        names = default_names

        if data_yaml_path.exists():
            try:
                import yaml

                with open(data_yaml_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
                if isinstance(existing.get("names"), list) and existing["names"]:
                    names = existing["names"]
                if isinstance(existing.get("nc"), int) and existing["nc"] > 0:
                    nc = existing["nc"]
                else:
                    nc = len(names)
                print("   ℹ️  Reusing classes from existing data.yaml")
            except Exception as e:
                print(f"   ⚠️  Could not parse existing data.yaml ({e}), inferring classes from labels")
                existing = {}
        else:
            existing = {}

        classes = set()
        if self.labels_dir.exists():
            for label_file in self.labels_dir.rglob("*.txt"):
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            classes.add(int(parts[0]))
                        except ValueError:
                            continue

        if classes:
            inferred_nc = max(classes) + 1
            if inferred_nc > 0:
                nc = inferred_nc
                if len(names) != nc:
                    names = [f"class_{i}" for i in range(nc)]

        data_yaml_content = {
            "path": str(self.project_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": nc,
            "names": names,
        }

        try:
            import yaml

            with open(data_yaml_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data_yaml_content, f, sort_keys=False, allow_unicode=True)
        except Exception:
            with open(data_yaml_path, "w", encoding="utf-8") as f:
                f.write(f"path: {self.project_dir}\n")
                f.write("train: images/train\n")
                f.write("val: images/val\n")
                f.write("test: images/test\n\n")
                f.write(f"nc: {nc}\n")
                f.write(f"names: {names}\n")

        print(f"✅ Created data.yaml with {nc} classes")
        return data_yaml_path

    def train_yolo(self, epochs=10, batch_size=16, quick=False):
        """Train YOLO model"""
        print("\n" + "="*60)
        print("🎯 Training YOLO Model...")
        print("="*60)
        
        try:
            from ultralytics import YOLO
            
            # Reduce for quick mode
            if quick:
                epochs = 3
                batch_size = 8
            
            print(f"   Epochs: {epochs}")
            print(f"   Batch Size: {batch_size}")
            print(f"   Quick Mode: {quick}")
            
            # Create data yaml
            data_yaml = self.create_data_yaml()
            
            # Initialize YOLO
            model = YOLO('yolov8m.pt')  # medium model

            device = 0
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
            
            # Train
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=416,
                batch=batch_size,
                patience=3,
                device=device,
                project=str(self.project_dir / "runs"),
                name="detect",
                exist_ok=True,
                verbose=True
            )
            
            print("✅ Training completed!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    def run_inference(self, source=None, confidence=0.5):
        """Run inference on images"""
        print("\n" + "="*60)
        print("🔍 Running Inference...")
        print("="*60)
        
        try:
            from ultralytics import YOLO
            
            # Find best model
            model_candidates = [
                self.project_dir / "runs" / "detect" / "weights" / "best.pt",
                self.project_dir / "runs" / "detect" / "train" / "weights" / "best.pt",
                self.project_dir / "runs" / "train" / "weights" / "best.pt",
            ]
            best_model = next((m for m in model_candidates if m.exists()), None)

            if best_model is None:
                print("❌ Best model not found in expected paths:")
                for candidate in model_candidates:
                    print(f"   - {candidate}")
                return False
            
            model = YOLO(str(best_model))
            
            # Set source
            if source is None:
                source = str(self.images_dir / "test")
            
            print(f"   Source: {source}")
            print(f"   Confidence: {confidence}")
            
            # Run inference
            results = model.predict(
                source=source,
                conf=confidence,
                project=str(self.project_dir / "runs"),
                name="predict",
                exist_ok=True,
                save=True,
                save_txt=True
            )
            
            print("✅ Inference completed!")
            print(f"   Results saved to: {self.project_dir / 'runs' / 'predict'}")
            return True
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False

    def create_submission(self):
        """Create submission.csv"""
        print("\n" + "="*60)
        print("📊 Creating Submission CSV...")
        print("="*60)
        
        try:
            import pandas as pd
            from pathlib import Path
            
            # Read predictions
            pred_dir = self.project_dir / "runs" / "predict" / "labels"
            
            if not pred_dir.exists():
                print("❌ No predictions found!")
                return False
            
            submissions = []
            
            for txt_file in pred_dir.glob("*.txt"):
                image_name = txt_file.stem
                
                with open(txt_file) as f:
                    detections = f.read().strip()
                
                submissions.append({
                    'image_id': image_name,
                    'predictions': detections
                })
            
            # Save to CSV
            df = pd.DataFrame(submissions)
            csv_path = self.project_dir / "submission.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Submission created: {csv_path}")
            print(f"   Total predictions: {len(submissions)}")
            return csv_path
        except Exception as e:
            print(f"❌ Submission creation failed: {e}")
            return False

    def check_setup(self):
        """Check if all required files exist"""
        print("\n" + "="*60)
        print("✅ Checking Setup...")
        print("="*60)
        
        checks = {
            "requirements.txt": self.project_dir / "requirements.txt",
            "images/train": self.images_dir / "train",
            "labels/train": self.labels_dir / "train",
        }
        
        all_ok = True
        for check_name, path in checks.items():
            exists = path.exists()
            status = "✅" if exists else "❌"
            print(f"   {status} {check_name}: {path}")
            if not exists:
                all_ok = False
        
        return all_ok

    def run(self, args):
        """Main execution pipeline"""
        print("\n" + "🚀 " * 20)
        print("DAL SHEMAGH DETECTION CHALLENGE")
        print("🚀 " * 20)
        
        # Step 1: Setup Kaggle
        if not self.setup_kaggle_api():
            return False
        
        # Step 2: Install requirements
        if not self.install_requirements():
            return False
        
        # Step 3: Download data (if requested)
        if args.download:
            if not self.download_data():
                print("⚠️  Download failed, continuing without download...")

        # Align extracted Kaggle layout even when --download is skipped
        self.prepare_dataset_layout()
        
        # Check setup
        if not self.check_setup():
            print("⚠️  Some files are missing")
        
        # Step 4: Train (unless inference-only)
        if not args.inference_only:
            if not self.train_yolo(
                epochs=10 if not args.quick else 3,
                batch_size=16 if not args.quick else 8,
                quick=args.quick
            ):
                return False
        
        # Step 5: Inference (unless train-only)
        if not args.train_only:
            if not self.run_inference():
                print("⚠️  Inference failed")
            else:
                # Step 6: Create submission
                self.create_submission()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📁 Results Directory: {self.project_dir / 'runs'}")
        print(f"📊 Submission File: {self.project_dir / 'submission.csv'}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="🚀 DAL Shemagh Detection - YOLO Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --download --quick
  python run_all.py --quick
  python run_all.py --train-only
  python run_all.py --inference-only
  python run_all.py --download
        """
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick mode (3 epochs, batch=8)"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training, skip inference"
    )
    
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference, skip training"
    )
    
    parser.add_argument(
        "--project-dir",
        default=Path.cwd(),
        help="Project directory path"
    )
    
    args = parser.parse_args()
    
    # Create project instance
    project = DALProject(args.project_dir)
    
    # Run pipeline
    success = project.run(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
