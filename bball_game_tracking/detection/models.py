from ultralytics import YOLO
import torch
import cv2
from typing import List
from tqdm import tqdm
from ..detection.detection import Detection
from ..utils.video import calculate_frame_skip

def process_video_with_model(model_path: str, video_path: str, model_id: int,
                           max_memory: float, stream: torch.cuda.Stream) -> List[Detection]:
    """
    Process video with model using specified GPU memory partition and CUDA stream
    """
    with torch.cuda.stream(stream):
        model = YOLO(model_path)
        # Force model to use only allocated memory
        for param in model.parameters():
            param.pin_memory()

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        rounded_fps = round(fps / 5) * 5
        frame_skip = calculate_frame_skip(rounded_fps)

        all_detections = []

        model_names = ["  Player  ", "Basketball"]  # Assign custom model names here

        processed_frames = total_frames // (frame_skip + 1)

        with tqdm(total=total_frames, desc=f"Processing {model_names[model_id]}", position=model_id) as pbar:
            frame_idx = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                # Skip frames based on calculated frame_skip
                if frame_idx % (frame_skip + 1) != 0:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Process frame
                resized_frame = cv2.resize(frame, (640, 640))

                # Ensure processing happens in the assigned stream
                # with torch.cuda.stream(stream):
                results = model(resized_frame, verbose=False)

                # Convert detections to our format
                for det in results[0].boxes.data.tolist():
                    x1, y1, x2, y2, conf, class_id = det
                    class_name = model.model.names[int(class_id)]
                    detection = Detection(
                        frame_idx=frame_idx,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name,
                        model_id=model_id
                    )
                    all_detections.append(detection)

                frame_idx += 1
                pbar.update(1)

                # Synchronize stream to prevent memory overflow
                # stream.synchronize()

        video.release()
        return all_detections

def process_video_with_model_cpu(model_path: str, video_path: str, model_id: int,
                               num_cores: int) -> List[Detection]:
    """
    Process video with model using CPU cores
    """
    # Force model to use CPU
    model = YOLO(model_path)
    model.cpu()

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    rounded_fps = round(fps / 5) * 5
    frame_skip = calculate_frame_skip(rounded_fps)

    all_detections = []
    model_names = ["  Player  ", "Basketball"]

    with tqdm(total=total_frames, desc=f"Processing {model_names[model_id]}", position=model_id) as pbar:
        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            resized_frame = cv2.resize(frame, (640, 640))

            # Process frame on CPU
            results = model(resized_frame, verbose=False)

            for det in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = det
                class_name = model.model.names[int(class_id)]
                detection = Detection(
                    frame_idx=frame_idx,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(class_id),
                    class_name=class_name,
                    model_id=model_id
                )
                all_detections.append(detection)

            frame_idx += 1
            pbar.update(1)

    video.release()
    return all_detections
