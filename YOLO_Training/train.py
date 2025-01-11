from ultralytics import YOLO
import os

# Ensure the working directory is set correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

data_yaml_path = os.path.join(script_dir, "_annotations/data.yaml")

# Initialize pretrained model
model = YOLO("yolov8n.pt")

results = model.train(
    data=data_yaml_path,
    epochs=100,
    project="runs",
    name="b_nano_model",
    device=0,
    patience=15,
    imgsz=640
)
