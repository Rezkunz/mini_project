from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    # Note: For demonstration/testing, we use 5 epochs. In a real scenario, use 50-100 epochs.
    results = model.train(
        data="dataset/data_indo.yaml", 
        epochs=50, 
        imgsz=640,
        batch=8, # Kurangi batch size agar tidak out of memory
        project="car_plate_detection",
        name="yolov8_plate_indo",
        device=0 # Menggunakan GPU pertama
    )
    
    print("Training finished!")
    return results

if __name__ == "__main__":
    train_model()
