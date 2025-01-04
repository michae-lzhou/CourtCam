from ultralytics import YOLO
import os

# Ensure the working directory is set correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

model = YOLO("yolov8m.yaml")
# model = YOLO("yolo_dataset/checkpoints/medium-Model/weights/best.pt")
data_yaml_path = os.path.join(script_dir, "data.yaml")

# Train the model with explicit dataset path
results = model.train(
    data=data_yaml_path,
    epochs=500,
    project="dataset/checkpoints",
    name="bhp_medium_model",
    batch=64,
    device="0",
    patience=40,
    imgsz=640,
    verbose=True,
    val=True
)
