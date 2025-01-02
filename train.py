from ultralytics import YOLO
import os

# Ensure the working directory is set correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

model = YOLO("yolov5n.yaml")
data_yaml_path = os.path.join(script_dir, "data.yaml")

# Train the model with explicit dataset path
results = model.train(
    data=data_yaml_path,
    epochs=100,
    project="dataset/checkpoints",
    name="small-Model",
    batch=32,
    device="cpu",
    patience=40,
    imgsz=640,
    verbose=True,
    val=True
)
