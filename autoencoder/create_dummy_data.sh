#!/bin/bash
# Set the base data directory
DATA_DIR="./data"

# Create the base data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Create 27 folders and populate them with random NumPy files
for i in $(seq 1 5); do
    # Create a folder named folder_<i>
    FOLDER_NAME="$DATA_DIR/folder_$i"
    mkdir -p "$FOLDER_NAME"
    
    # Create 20 random NumPy files in the folder
    for j in $(seq 1 20); do
        # Generate a random numpy array of shape (2, 2, 60, 256, 256)
        FILE_NAME="$FOLDER_NAME/file_$j.npy"
        
        # Use Python to generate the random NumPy array and save it
        python -c "import numpy as np; np.save('$FILE_NAME', np.random.rand(2, 2, 60, 256, 256).astype(np.float16) * 20000)"
    done
done
