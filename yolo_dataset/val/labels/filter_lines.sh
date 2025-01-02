#!/bin/zsh

# Loop through all .txt files in current directory
for file in *.txt; do
    # Check if file exists (in case no .txt files found)
    echo "Processing file $file"
    if [[ -f $file ]]; then
        # Create a temporary file
        temp_file=$(mktemp)
        
        # Filter lines that start with 0 and write to temp file
        grep '^0' "$file" > "$temp_file" || true
        
        # Replace original file with filtered content
        mv "$temp_file" "$file"
    fi
done
