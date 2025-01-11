#!/bin/bash
# This script removes all lines starting with "0" from all .txt files in the current directory.

# Loop through all .txt files in the current directory
for file in *.txt; do
  # Check if the file exists
  if [ -f "$file" ]; then
    echo "Processing $file..."
    # Replace first character "1" with "0" in lines starting with "1"
    sed -i 's/^2/0/' "$file"
    # Replace first character "2" with "1" in lines starting with "2"
    
    # sed -i 's/^2/1/' "$file"
    echo "First '2' replaced with '0' in $file."
  fi
done
