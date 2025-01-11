#!/bin/bash

# Download all of the requirements to run the program
pip install -r bball_game_tracking/requirements.txt > /dev/null 2>&1

# Find all .mp4 files in the current directory
mp4_files=(*.mp4)

# Check if exactly one .mp4 file exists
if [ ${#mp4_files[@]} -eq 1 ]; then
  # Open a new terminal window and run the Python script
  if command -v gnome-terminal &> /dev/null; then
    gnome-terminal -- bash -c "python3 -m bball_game_tracking.main '${mp4_files[0]}'; echo 'Press any key to exit'; read -n 1"
  elif command -v xterm &> /dev/null; then
    xterm -hold -e "python3 -m bball_game_tracking.main '${mp4_files[0]}'; echo 'Press any key to exit'; read -n 1"
  elif command -v open &> /dev/null; then
    # For macOS
    osascript -e 'tell application "Terminal" to do script "python3 -m bball_game_tracking.main '${mp4_files[0]}'; echo 'Press any key to exit'; read -n 1"'
  else
    echo "No compatible terminal found."
  fi
else
  echo "Please ensure there is exactly one .mp4 file in the current directory and try again"
  echo 'Press any key to exit'
  read -n 1
fi
