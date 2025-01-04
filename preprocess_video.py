import cv2
import os
import random
from tqdm import tqdm

def preprocess():
    # Set up output folder
    output_folder = "_raw_frames"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all videos in _videos folder
    video_folder = "_videos"
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    if not video_files:
        print("No .mp4 files found in _videos folder!")
        return
        
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        vid_name = os.path.splitext(video_file)[0]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_file}")
            continue
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_extract = min(1000, total_frames)
        
        # Generate random frame indices
        frame_indices = sorted(random.sample(range(total_frames), frames_to_extract))
        
        print(f"\nProcessing {video_file}...")
        
        # Create progress bar
        progress_bar = tqdm(total=frames_to_extract,
                          desc=f"Extracting frames from {vid_name}",
                          unit="frames")
        
        frame_count = 0
        current_frame = 0
        
        for target_frame in frame_indices:
            # Skip frames until we reach the next target frame
            while current_frame < target_frame:
                ret = cap.grab()  # Just grab the frame without decoding
                if not ret:
                    break
                current_frame += 1
            
            # Read the target frame
            ret, frame = cap.retrieve()  # Decode only the frame we want
            if not ret:
                break
                
            # Save frame using the original frame number in the filename
            frame_filename = os.path.join(output_folder, f"{vid_name}_frame_{target_frame:06d}.jpg")
            frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(frame_filename, frame)
            
            frame_count += 1
            current_frame += 1
            progress_bar.update(1)
        
        # Cleanup
        progress_bar.close()
        cap.release()
        print(f"✓ Completed {vid_name}: extracted {frame_count} frames")

if __name__ == "__main__":
    preprocess()
