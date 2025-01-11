# CourtCam

Ever dreamed of never touching your tripod again during your son/daughter's basketball game, but still want to capture all of the action? Introducing **CourtCam**, a simple, cross-platform Python-based application designed to automatically detect and track action in video footage. With the power of computer vision and machine learning, it processes game footage to provide enhanced video capture, ensuring that you never miss a shot — all without needing to adjust your camera manually.

Simply **(1)** record your child's basketball game on a tripod with a 0.5 ultra-wide lens shot from mid-court, **(2)** place the .mp4 file in the same folder, and **(3)** right-click on `run_tracking.sh` to "Run as a Program." The application will take care of the rest.

## Why Should You Use It?

- **Simple to Use**: No need for complex setup — just place the video in the folder and run the script!
- **Cross-Platform**: Works on Windows, macOS, and Linux.
- **Lightweight**: Doesn't require heavy resources and runs efficiently even on basic hardware. (Bonus if you have a GPU with CUDA!)
- **Fast Processing**: The program guarantees finishing processing in at most **1.5x the original video's length**, ensuring you get your enhanced video quickly without waiting for hours.

## How to Use

### Action Tracking

To analyze a basketball game video, follow these steps:

1. **Download the Repository**  
   Choose one of the following methods to get the repository files:  

   - **Using Git (Recommended):**  
     Open a terminal and run the following command:  
     ```bash
     git clone https://github.com/michae-lzhou/bball-game-analyst.git
     ```  

   - **Without Git:**  
     Navigate to the repository on GitHub, click the green **Code** button, and select **Download ZIP**. Extract the ZIP file after downloading.
    
2. **Set Up Dependencies (Optional)**  
   Follow the installation guide below to ensure all necessary dependencies are installed.  

3. **Prepare the CourtCam Folder**  
   Ensure that the **CourtCam** folder contains the following:  
   - `bball_game_tracking` folder  
   - `run_tracking.sh`  
   - `[YOUR_VIDEO.mp4]`  

4. **Add Your Video**  
   Drag and drop your `.mp4` video into the **CourtCam** folder.  

5. **Run the Program**  
   Right-click on **run_tracking.sh** and select **"Run as a Program"**.  

6. **Follow the Instructions**  
   Complete the steps in the pop-up menu and terminal to initiate the program.  

7. **Important Reminder**  
   Ensure there is only one `.mp4` file in the folder at a time to avoid conflicts.  

**Note:** You can exit the program at any time by pressing `CTRL-C` a couple of times. However, please be aware that this might result in incomplete processing or corrupted outputs.

## Installation

### Prerequisites

- Python 3.6+
- Pip (Python package manager)

**Install Dependencies**
1.  It is recommended to use a virtual environment. You can set it up as follows:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

 2. Install the required dependencies (Optional: Handled when ran):
    ```bash
    pip install -r bball_game_tracking/requirements.txt
    ```

**Primary Libraries Used:**
    - Ultralytics (YOLO)
    - OpenCV
    - NumPy
    - scikit-learn

## Features

- **Basketball Tracking**: Automatically detects and tracks basketballs in real-time using a YOLO-based object detection model.
- **Frame Cropping**: Crops video frames around the detected basketballs for a more focused and engaging viewing experience.
- **Video Processing**: Generates output videos with real-time tracking and enhanced frames.
- **Super Resolution (Coming Soon)**: Smoothens video playback and increases video quality from HD to 4k with AI.

## Expected Metrics

### Basketball Game Tracker Performance

Please note that performance may vary due to limited training data (especially during free-throws and time-outs)

### Processing Device

- **GPU (~7x speed)**: For optimal performance, we recommend using a GPU, as it significantly accelerates processing time. Ensure CUDA is enabled for best results.
- **CPU (~0.85x speed)**: While slower, CPU processing is universally compatible and does not require specialized hardware.

### Quality Settings

- **Original**: Processes the video at full quality. The runtime scales with the original video's frame rate (FPS).
- **Fast (30 FPS)**: Offers faster processing with a fixed video quality of 30 FPS. This is ideal when speed is prioritized over quality.

### Notes
- If GPU processing encounters an issue, the program will automatically fall back to CPU processing, ensuring compatibility without additional configuration.
