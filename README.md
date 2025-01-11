# CourtCam

Ever dreamed of never touching your tripod again during your son/daughter's basketball game, but still want to capture all of the action? Introducing **CourtCam**, a simple, cross-platform Python-based application designed to automatically detect and track action in video footage. With the power of computer vision and machine learning, it processes game footage to provide enhanced video capture, ensuring that you never miss a shot — all without needing to adjust your camera manually.

Simply **(1)** record your child's basketball game on a tripod with a 0.5 ultra-wide lens shot from mid-court, **(2)** place the .mp4 file in the same folder, and **(3)** right-click on `run_tracking.sh` to "Run as a Program." The application will take care of the rest.

## Demo


https://github.com/user-attachments/assets/7a41935e-2ef5-4e78-8fb4-28db2b5c0d1b



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
     git clone https://github.com/michae-lzhou/CourtCam.git
     ```  

   - **Without Git:**  
     Navigate to the repository on GitHub, click the green **Code** button, and select **Download ZIP**. Extract the ZIP file after downloading.
    
2. **Set Up Dependencies (Optional)**  
   Follow the installation guide below to ensure all necessary dependencies are installed.  

3. **Prepare the CourtCam Folder**  
   Ensure that the **CourtCam** folder contains the following:  
   - `bball_game_tracking` folder
   - `README.md`
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

**Note:** You can exit the program at any time by pressing `CTRL-C` a couple of times. However, please be aware that this might result in incomplete processing or corrupted outputs. Restore the folder to step 3. and try again.

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

- **GPU (~7x speed)**: For optimal performance, I recommend using a GPU, as it significantly accelerates processing time. Ensure CUDA is enabled for best results.
- **CPU (~0.85x speed)**: While slower, CPU processing is universally compatible and does not require specialized hardware.

### Quality Settings

- **Original**: Processes the video at full quality. The runtime scales with the original video's frame rate (FPS).
- **Fast (30 FPS)**: Offers faster processing with a fixed video quality of 30 FPS. This is ideal when speed is prioritized over quality.

### Notes
- If GPU processing encounters an issue, the program will automatically fall back to CPU processing, ensuring compatibility without additional configuration.


## References
Huge shoutouts to these amazing resources on the niche problems I was trying to solve! (That you'll also likely run into if you want to do something similar)

1. **[How to Detect Basketball Game using Deep Learning](https://www.youtube.com/watch?v=i8k8YP0oy00)**
   - By Eran Feit, this is the amazing tutorial I started with to train my YOLO models! The annotation part with GroundingDINO saved me a lot of headache and late nights drawing my own bounding boxes.

2. **[How to Automatically Remove Fish-Eye(Wide Angle) Lens Distortion for Any Camera](https://www.youtube.com/watch?v=MAoQqhcKKAo)**
   - By WalkWithMe, I didn't realize that iPhones already have built-in distortion removal when taking videos from ultra-wide lens, but this tutorial is extremely helpful for people using specialized recording devices that don't have this built-in functionality.

3. **[A simple way of creating a custom object detection model](https://towardsdatascience.com/chess-rolls-or-basketball-lets-create-a-custom-object-detection-model-ef53028eac7d)**
   - By Piotr Skalski, an end-to-end guide on building a custom object detection model with detailed steps and examples.

4. **[makesense.ai](https://www.makesense.ai/)**
   - From Reference 3, ONE OF THE BEST TOOLS I'VE USED TO HELP ME LABEL DATA. The web-interface makes everything so much easier, and the process is so streamlined for labeling your own custom data.
  
5. **[Extending YOLOv8 COCO Model With New Classes Without Affecting Old Weights](https://y-t-g.github.io/tutorials/yolov8n-add-classes/)**
   - By Mohammed Yasin, although I ended up not using the merged model due to the low accuracy, but this guide is exactly what you are looking for when you want to "run two YOLO models for the price of one". Mohammed was extremely responsive too, and answered my questions completely within 12 hours.
