#!/bin/bash
# Check if a file is passed as an argument
if [ -f "$1" ]; then
  # Run your Python script with the video file
  python3 -m bball_game_tracking.main "$1"
else
  echo "Please drag and drop a video file."
fi

