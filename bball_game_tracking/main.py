import cv2
import ffmpeg
import os
import torch
import torch.cuda
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tkinter import messagebox
from typing import List

from .detection.models import process_video_with_model, process_video_with_model_cpu
from .utils.video import create_output_video
from .config.menu import show_configuration_menu

def extract_audio(input_file, output_audio_file):
    ffmpeg.input(input_file).output(output_audio_file, vn=None, acodec='copy').run()

def re_add_audio(input_video, input_audio, output_video):
    ffmpeg.input(input_video).input(input_audio).output(output_video, c='copy', acodec='aac', strict='experimental').run()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument('video_file', type=str, help="Path to the video file to process.")
    return parser.parse_args()

def setup_gpu_partition(gpu_id: int, max_memory: float):
    """Set up GPU memory partition"""
    torch.cuda.set_device(gpu_id)
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
    max_allocated = int(total_memory * max_memory)
    torch.cuda.set_per_process_memory_fraction(max_memory)
    return torch.cuda.Stream()

def setup_processing_device(model_paths: List[str], memory_fractions: List[float]):
    """Sets up either GPU streams or CPU cores based on user choice"""
    use_cuda = False

    if memory_fractions:
        try:
            if not torch.cuda.is_available():
                messagebox.showerror(
                    "CUDA Error",
                    "CUDA is not available despite selecting GPU processing.\n" +
                    "Falling back to CPU processing."
                )
            else:
                use_cuda = True
                streams = [setup_gpu_partition(0, mf) for mf in memory_fractions]
                return True, streams
        except Exception as e:
            messagebox.showerror(
                "CUDA Error",
                f"Error setting up CUDA: {str(e)}\nFalling back to CPU processing."
            )

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    cores_per_model = max(1, num_cores // len(model_paths))
    return False, cores_per_model

def run_partitioned_detection(video_path: str, model_paths: List[str],
                            memory_fractions: List[float], output_path: str):
    """Run models with either GPU or CPU processing"""
    use_cuda, processing_resources = setup_processing_device(model_paths, memory_fractions)
    
    if use_cuda:
        with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
            futures = [
                executor.submit(
                    process_video_with_model,
                    model_path,
                    video_path,
                    model_id,
                    memory_fractions[model_id],
                    processing_resources[model_id]
                )
                for model_id, model_path in enumerate(model_paths)
            ]
            
            all_detections = []
            for future in futures:
                all_detections.extend(future.result())
    else:
        with ThreadPoolExecutor(max_workers=len(model_paths)) as executor:
            futures = [
                executor.submit(
                    process_video_with_model_cpu,
                    model_path,
                    video_path,
                    model_id,
                    processing_resources
                )
                for model_id, model_path in enumerate(model_paths)
            ]
            
            all_detections = []
            for future in futures:
                all_detections.extend(future.result())
    
    return all_detections

def main():
    args = parse_arguments()  # Get the arguments
    video_file = args.video_file  # The path of the video file

    output_audio = 'audio.aac'
    extract_audio(video_file, output_audio)

    # Show configuration menu
    config = show_configuration_menu()

    # Set up processing based on configuration
    use_cuda = config['processing_device'] == 'GPU'
    fast_mode = config['quality'] == 'Fast'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    player_model_path = os.path.join(BASE_DIR, '_runs', 'hp_nano_model', 'weights', 'best.pt')
    basketball_model_path = os.path.join(BASE_DIR, '_runs', 'b_nano_model', 'weights', 'best.pt')

    model_paths = [
        # 'runs/hp_nano_model/weights/best.pt',  # Player model
        # 'runs/b_nano_model/weights/best.pt'    # Basketball model
        player_model_path,
        basketball_model_path
    ]

    # Allocate 40% of GPU memory to each model if using GPU
    memory_fractions = [0.4, 0.4] if use_cuda else None


    # video_file = os.path.join(BASE_DIR, '_videos', 'jack_brandeis.mp4')
    output_file = 'output.mp4'

    video = cv2.VideoCapture(video_file)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    rounded_fps = round(fps / 5) * 5
    print(f"Processing {video_file}")
    print(f"Video FPS: {rounded_fps}")
    video.release()

    print()
    print("\033[1mStep 1: Information Extraction\033[0m")
    detections = run_partitioned_detection(video_file, model_paths, memory_fractions, output_file)

    print()
    print("\033[1mStep 2: Creating Output Video\033[0m")
    print(f"Output FPS: {30 if fast_mode else rounded_fps}")
    model_names = ["Player_Model", "Basketball_Model"]
    create_output_video(video_file, detections, model_names, output_file,
                       force_30fps=fast_mode)

    file_name_without_extension = video_file.replace('.mp4', '')
    final_name = file_name_without_extension.split('/')[-1]

    re_add_audio('output.mp4', 'audio.aac', f'[MOD]{final_name}.mp4')

    os.remove('output.mp4')
    os.remove('audio.aac')

if __name__ == '__main__':
    main()
