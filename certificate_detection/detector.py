# certificate_detection/detector.py
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

class CertificateDetector:
    def __init__(self, model_path="./certificate_detection/yolov8/best.pt", confidence_threshold=0.5):
        """
        Initialize the certificate detector with your trained model
        
        Args:
            model_path: Path to your trained best.pt model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        
        # Load the model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            print(f"Model not found at {self.model_path}")
            print("Please make sure your best.pt file is in the certificate_detection folder")
    
    def load_model(self):
        """Load your trained YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Certificate detection model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_and_crop_certificates(self, image_path, output_folder="./processed_certificates", padding=10):
        """
        Detect certificates in image and crop them
        
        Args:
            image_path: Path to input image
            output_folder: Folder to save cropped certificates
            padding: Padding around detected certificate
            
        Returns:
            List of paths to cropped certificate images
        """
        if self.model is None:
            print("Model not loaded. Cannot perform detection.")
            return []
        
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return []
            
            os.makedirs(output_folder, exist_ok=True)
            
            cropped_paths = []
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        
                        print(f"Detected certificate with confidence: {confidence:.2f}")
                        
                        # Add padding
                        h, w = image.shape[:2]
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)
                        
                        # Crop certificate
                        cropped_cert = image[y1:y2, x1:x2]
                        
                        # Save cropped certificate
                        output_filename = f"{base_filename}_detected_{i+1}.png"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        cv2.imwrite(output_path, cropped_cert)
                        cropped_paths.append(output_filename)  # Just return filename for consistency
                        
                        print(f"Cropped certificate saved: {output_path}")
            
            return cropped_paths
        
        except Exception as e:
            print(f"Error during detection and cropping: {e}")
            return []