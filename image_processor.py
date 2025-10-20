import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import torch
import torch.nn.functional as F
import os
import argparse # Import argparse for command-line arguments
# How to run
#python image_processor.py /media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy --no_gui

'''
How to Run from the Command Line:

python 4d_image_processor.py /media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy --no_gui

This will process the specified image, create the output folder and .npy movies, and then exit without showing the GUI. The output folder will be created in the current directory (.) by default.

Processing and Saving to a Specific Output Directory:
python 4d_image_processor.py /media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy --output_dir /path/to/your/output/folder --no_gui


Processing without original channel saves: python 4d_image_processor.py /media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy --no_save_original --no_gui


With GUI (and processing in the background): python 4d_image_processor.py /media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000059.npy


This will first process and save the Z-slice movies (as in option 2, using the default output directory), and then launch the interactive GUI viewer.
'''

def normalize_and_mask(img_slice_2d, eps=1e-6, mask_threshold=0.1):
    """
    Remove morphology-related brightness (TMRM) from MitoTracker
    to emphasize functional membrane potential signal.

    Args:
        img_slice_2d: A 2D slice from the image, expected shape [C, H, W] or [C, W, H]
             channel 0 = TMRM (morphology)
             channel 1 = MitoTracker (function + morphology)
        eps: small value to prevent divide-by-zero
        mask_threshold: relative threshold for mitochondrial mask
    Returns:
        functional_masked: The processed 2D image showing functional signal.
        mask: The binary mitochondrial mask.
    """
    # Ensure img_slice_2d has a channel dimension at the beginning
    if img_slice_2d.ndim == 2: # Assuming single channel if 2D
        tmrm = img_slice_2d
        mitotk = img_slice_2d # Fallback if only one channel is provided, though function expects 2
    elif img_slice_2d.ndim == 3 and img_slice_2d.shape[0] >= 2: # Expected [C, H, W]
        tmrm = img_slice_2d[0, :, :]      # morphology only
        mitotk = img_slice_2d[1, :, :]    # morphology + function
    else:
        # Handle cases where input might not perfectly match expected [C, H, W]
        # For simplicity, if less than 2 channels, use the first channel for both
        print("Warning: normalize_and_mask received unexpected input shape. Using first channel for both TMRM and MitoTracker.")
        tmrm = img_slice_2d[0, :, :]
        mitotk = img_slice_2d[0, :, :]


    # Convert to float32 for calculations if not already
    tmrm = tmrm.astype(np.float32)
    mitotk = mitotk.astype(np.float32)

    # --- Step 1: background subtraction ---
    tmrm_min = tmrm.min()
    mitotk_min = mitotk.min()
    if tmrm_min < 0: # Only subtract if there are negative values
        tmrm = tmrm - tmrm_min
    if mitotk_min < 0:
        mitotk = mitotk - mitotk_min

    # --- Step 2: normalize both to [0, 1] ---
    tmrm_max = tmrm.max()
    mitotk_max = mitotk.max()
    if tmrm_max > 0:
        tmrm = tmrm / (tmrm_max + eps)
    if mitotk_max > 0:
        mitotk = mitotk / (mitotk_max + eps)

    # --- Step 3: remove morphology contribution ---
    # Divide MitoTracker by TMRM to isolate functional signal
    functional = mitotk / (tmrm + eps)
    
    functional_max = functional.max()
    if functional_max > 0:
        functional = functional / (functional_max + eps)
    
    # --- Step 4: create mitochondrial mask ---
    # Use the *original* (background-subtracted, normalized) tmrm for masking
    mask = (tmrm > mask_threshold).astype(np.float32) # Already normalized, so use raw threshold

    # --- Step 5: apply mask ---
    functional_masked = functional * mask

    return functional_masked, mask


def process_and_save_z_movies(image_filepath, output_dir, save_original_z_slices=True):
    """
    Loads a 4D (time, channel, z, y, x) image, processes it, and saves
    movies for each z-slice as individual .npy files in a new directory.

    Args:
        image_filepath (str): Path to the .npy file containing the 4D image data.
        output_dir (str): The base directory where the output folder will be created.
        save_original_z_slices (bool): If True, also save the original channel data
                                         for each z-slice.
    """
    try:
        image_data_raw = np.load(image_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {image_filepath}")
        return
    except Exception as e:
        print(f"Error loading image data: {e}")
        return

    # Assuming data shape is (time_points, channels, z_slices, y_dim, x_dim)
    # The original code's comment said (time, z, y, x, channels) but the usage implied (time, channel, z, y, x)
    # Let's explicitly check and reorder if necessary for consistency with my assumption
    if image_data_raw.ndim != 5:
        print(f"Error: Expected 5 dimensions, but got {image_data_raw.ndim}. Please check data shape.")
        return

    # Assuming input is (T, C, Z, H, W)
    # If your data is actually (T, Z, H, W, C), you'd need to reorder:
    # image_data_raw = np.transpose(image_data_raw, (0, 4, 1, 2, 3)) # (T, C, Z, H, W)
    # Let's assume the provided format (20,2,60,256,256) which is (T, C, Z, H, W)
    image_data = image_data_raw 

    time_points, num_channels, z_slices, y_dim, x_dim = image_data.shape

    if num_channels < 2:
        print(f"Warning: Expected at least 2 channels for processing, but found {num_channels}. "
              "Functional signal calculation might not be meaningful.")
        return

    # Create output directory
    image_filename_base = os.path.splitext(os.path.basename(image_filepath))[0]
    output_folder = os.path.join(output_dir, f"{image_filename_base}_processed_movies")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")

    # Process and save each z-slice
    for z_idx in range(z_slices):
        print(f"Processing Z-slice {z_idx + 1}/{z_slices}...")
        
        # Prepare arrays to store processed frames for this z-slice
        # Each frame will be (H, W) for functional_masked
        functional_movie_frames = np.zeros((time_points, y_dim, x_dim), dtype=np.float32)
        mask_movie_frames = np.zeros((time_points, y_dim, x_dim), dtype=np.float32)
        
        if save_original_z_slices:
            # For original channels, shape will be (T, C, H, W)
            original_channels_z_slice_movie = np.zeros((time_points, num_channels, y_dim, x_dim), dtype=image_data.dtype)

        for t_idx in range(time_points):
            # Get the current 2D slice for all channels: image_data[t_idx, :, z_idx, :, :] -> shape (C, H, W)
            current_slice_2d_all_channels = image_data[t_idx, :, z_idx, :, :]
            
            # Process this 2D slice
            functional_masked_frame, mask_frame = normalize_and_mask(current_slice_2d_all_channels)
            
            functional_movie_frames[t_idx] = functional_masked_frame
            mask_movie_frames[t_idx] = mask_frame

            if save_original_z_slices:
                original_channels_z_slice_movie[t_idx] = current_slice_2d_all_channels

        # Save the processed movies for this z-slice
        z_slice_folder = os.path.join(output_folder, f"Z_slice_{z_idx:03d}")
        os.makedirs(z_slice_folder, exist_ok=True)

        np.save(os.path.join(z_slice_folder, f"functional_masked_movie_z{z_idx:03d}.npy"), functional_movie_frames)
        np.save(os.path.join(z_slice_folder, f"mask_movie_z{z_idx:03d}.npy"), mask_movie_frames)
        
        if save_original_z_slices:
            np.save(os.path.join(z_slice_folder, f"original_channels_movie_z{z_idx:03d}.npy"), original_channels_z_slice_movie)
            
        print(f"Saved movies for Z-slice {z_idx} to {z_slice_folder}")

    print(f"✅ All Z-slice movies processed and saved to {output_folder}")


def view_4d_image_with_sliders(image_filepath, display_gui=True):
    """
    Views a 4D (time, channel, z, y, x) image from a .npy file with sliders
    for time and z-slice selection. Can be toggled off for command line only usage.

    Args:
        image_filepath (str): Path to the .npy file containing the 4D image data.
        display_gui (bool): If True, displays the interactive matplotlib GUI.
    """
    try:
        image_data = np.load(image_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {image_filepath}")
        return
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    
    # Assuming data shape is (time_points, channels, z_slices, y_dim, x_dim)
    if image_data.ndim != 5:
        print(f"Error: Expected 5 dimensions (time, channel, z, y, x), but got {image_data.ndim}")
        print("Please ensure your .npy file has the correct shape.")
        return

    time_points, num_channels, z_slices, y_dim, x_dim = image_data.shape

    if num_channels < 2:
        print(f"Warning: Expected at least 2 channels, but found {num_channels}. "
              "Functional signal calculation might not be meaningful.")

    if not display_gui:
        print("GUI display is disabled. No interactive viewer will be shown.")
        return # Exit if GUI is not desired

    # Initial display parameters
    initial_time = 0
    initial_z = 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 7)) # Adjust figsize for better viewing
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.25, top=0.9) # Adjust margins

    # Display initial images for both channels
    im1 = ax[0].imshow(image_data[initial_time,0, initial_z, :, :], cmap='gray', vmin=image_data[:,0,:,:,:].min(), vmax=image_data[:,0,:,:,:].max())
    ax[0].set_title(f"Channel 0 (TMRM)\n(Time: {initial_time}, Z: {initial_z})")
    ax[0].axis('off')

    im2 = ax[1].imshow(image_data[initial_time,1, initial_z, :, :], cmap='gray', vmin=image_data[:,1,:,:,:].min(), vmax=image_data[:,1,:,:,:].max())
    ax[1].set_title(f"Channel 1 (MitoTracker)\n(Time: {initial_time}, Z: {initial_z})")
    ax[1].axis('off')

    # Calculate initial functional_masked image
    current_slice_2d_all_channels = image_data[initial_time, :, initial_z, :, :]
    initial_functional_masked, _ = normalize_and_mask(current_slice_2d_all_channels)
    im3 = ax[2].imshow(initial_functional_masked, cmap='gray', vmin=0, vmax=1) # Normalized to [0,1]
    ax[2].set_title(f"Functional Signal\n(Time: {initial_time}, Z: {initial_z})")
    ax[2].axis('off')
    
    # Add colorbars for better interpretation
    fig.colorbar(im1, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)

    # Create slider axes
    ax_time = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
    ax_z = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')

    # Create sliders
    time_slider = Slider(ax_time, 'Time', 0, time_points - 1, valinit=initial_time, valstep=1)
    z_slider = Slider(ax_z, 'Z-Slice', 0, z_slices - 1, valinit=initial_z, valstep=1)

    def update(val):
        current_time = int(time_slider.val)
        current_z = int(z_slider.val)

        im1.set_data(image_data[current_time,0, current_z, :, :])
        ax[0].set_title(f"Channel 0 (TMRM)\n(Time: {current_time}, Z: {current_z})")

        if num_channels >= 2:
            im2.set_data(image_data[current_time, 1,current_z, :, :])
            ax[1].set_title(f"Channel 1 (MitoTracker)\n(Time: {current_time}, Z: {current_z})")
            
            current_slice_2d_all_channels = image_data[current_time, :, current_z, :, :]
            functional_masked_frame, _ = normalize_and_mask(current_slice_2d_all_channels)
            im3.set_data(functional_masked_frame)
            ax[2].set_title(f"Functional Signal\n(Time: {current_time}, Z: {current_z})")
        else:
            ax[1].set_title(f"Channel 1 (Not available)")
            im2.set_data(np.zeros_like(image_data[current_time, 0,current_z, :, :])) # Show black if no second channel
            ax[2].set_title(f"Functional Signal (Not available)")
            im3.set_data(np.zeros_like(image_data[current_time, 0,current_z, :, :])) # Show black if no second channel
            
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    z_slider.on_changed(update)

    # --- Add Save Movie Button ---
    ax_save = plt.axes([0.8, 0.9, 0.15, 0.05])
    btn_save = Button(ax_save, 'Save Movie', color='lightblue', hovercolor='skyblue')

    def save_movie_gui_handler(event):
        print("Preparing movie export...")

        mode = input("Animate over [time/z]? ").strip().lower()
        if mode not in ["time", "z"]:
            print("Invalid choice. Please type 'time' or 'z'.")
            return

        save_path_base = os.path.splitext(image_filepath)[0]
        # Use a default name and ensure it's saved in a specific folder
        default_movie_filename = f"{os.path.basename(save_path_base)}_{mode}_movie.mp4"
        movie_output_dir = os.path.join(os.path.dirname(image_filepath), f"{image_filename_base}_gui_movies")
        os.makedirs(movie_output_dir, exist_ok=True)
        save_path = os.path.join(movie_output_dir, default_movie_filename)

        fps = 5  # frames per second

        frames = range(time_points) if mode == "time" else range(z_slices)

        def animate(i):
            if mode == "time":
                t, z = i, int(z_slider.val)
            else:
                t, z = int(time_slider.val), i

            im1.set_data(image_data[t, 0, z, :, :])
            im2.set_data(image_data[t, 1, z, :, :])
            im3.set_data(normalize_and_mask(image_data[t, :, z, :, :])[0])

            ax[0].set_title(f"Ch0 (TMRM)\n(t={t}, z={z})")
            ax[1].set_title(f"Ch1 (MitoTracker)\n(t={t}, z={z})")
            ax[2].set_title(f"Functional Signal\n(t={t}, z={z})")
            return [im1, im2, im3]

        ani = animation.FuncAnimation(fig, animate, frames=frames, blit=False, repeat=False)

        try:
            print(f"Saving movie to: {save_path}")
            ani.save(save_path, fps=fps, writer='ffmpeg', dpi=150) # Increased DPI for better quality
            print(f"✅ Movie saved successfully: {save_path}")
        except Exception as e:
            print(f"❌ Error saving movie: {e}")
            print("Make sure FFmpeg is installed and accessible in your PATH (e.g., `sudo apt install ffmpeg` on Linux).")
            print("You might also need to install `imagemagick` for GIF support, or set the writer explicitly (e.g., `writer='ffmpeg'`).")

    btn_save.on_clicked(save_movie_gui_handler)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and view 4D medical images (.npy files).")
    parser.add_argument("image_filepath", type=str,
                        help="Path to the .npy file containing the 4D image data (T, C, Z, H, W).")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save processed Z-slice movies. Defaults to current directory.")
    parser.add_argument("--no_gui", action="store_true",
                        help="Do not display the interactive GUI viewer. Only perform command-line processing.")
    parser.add_argument("--no_save_original", action="store_true",
                        help="Do not save original channel movies for each Z-slice, only processed functional and mask.")
    
    args = parser.parse_args()

    # --- Run the processing and saving function ---
    process_and_save_z_movies(
        image_filepath=args.image_filepath,
        output_dir=args.output_dir,
        save_original_z_slices=not args.no_save_original
    )

    # --- Run the viewer (if not disabled) ---
    if not args.no_gui:
        print("\nDisplaying interactive viewer...")
        view_4d_image_with_sliders(args.image_filepath, display_gui=True)
    else:
        print("\nInteractive GUI viewer skipped (--no_gui flag is set).")