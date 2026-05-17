#!/bin/bash

# Define the target directory name
TARGET_DIR="all_the_multi-tier_runs"

# Create the target directory if it doesn't already exist
mkdir -p "$TARGET_DIR"

echo "Scanning for .sh and .log files referencing 'apply_septq_multitier.py'..."
echo "----------------------------------------------------------------------"

# Counter for copied files
copied_count=0

# Use find to locate all .sh and .log files recursively
# We loop through them using a while read block to safely handle spaces in filenames
find . -type f \( -name "*.sh" -o -name "*.log" \) | while read -r file; do
    
    # Skip files that are already inside the destination directory to avoid infinite loops
    if [[ "$file" == *"/$TARGET_DIR/"* ]]; then
        continue
    fi

    # Check if the file contains the target script name
    if grep -q "apply_septq_multitier.py" "$file"; then
        filename=$(basename "$file")
        extension="${filename##*.}"
        filename_no_ext="${filename%.*}"
        
        dest_file="$TARGET_DIR/$filename"
        
        # Handle filename collisions (e.g., if multiple folders have a "run.log")
        counter=1
        while [ -f "$dest_file" ]; do
            dest_file="$TARGET_DIR/${filename_no_ext}_${counter}.${extension}"
            ((counter++))
         Bernardo_counter
        done

        # Copy the file to the new destination
        cp "$file" "$dest_file"
        echo "Copied: $file -> $dest_file"
        ((copied_count++))
    fi
done

echo "----------------------------------------------------------------------"
echo "Done! Check the '$TARGET_DIR' folder."
