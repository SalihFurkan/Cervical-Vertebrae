from ultralytics import YOLO

model = YOLO("yolov8s.pt", verbose=True)  # Start with small model

model.train(
    data="datasets/data.yaml",  # Path to your YAML file
    epochs=50,                # Number of training epochs
    imgsz=1013,                # Image size (default 640x640)
    batch=8,                 # Batch size (adjust based on GPU memory)
    name="cvm_yolov8",      # Name for output directory,
    device='mps'
)

