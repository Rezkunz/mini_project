import os
import glob
from ultralytics import YOLO

def test_model():
    # Load the trained model
    weights_path = "runs/detect/car_plate_detection/yolov8_plate_indo/weights/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Model weights not found at {weights_path}")
        print("Please train the model first by running train.py")
        return
        
    model = YOLO(weights_path)
    
    # Run inference on the test dataset
    test_images = glob.glob("dataset/test/images/*.jpg")
    if not test_images:
        print("No test images found!")
        return
        
    # Test on a few images
    results = model.predict(source=test_images[:5], save=True, project="car_plate_detection", name="predict")
    print("Inference finished. Results saved to car_plate_detection/predict")

if __name__ == "__main__":
    test_model()
