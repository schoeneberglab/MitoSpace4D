import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch
import os
import torch.nn.functional as F
from matplotlib.widgets import Button
import matplotlib.animation as animation

def load_folder_label_maps(drugs_to_labels_path = f"/home/mayunagupta/experiments/MitoSpace4D/extraction_utils/drugs_to_labels.txt"):
    # Load drug to label mappings
    drug_to_labels_dict = {}
    folder_to_label = {}
    folder_to_drug = {}
    with open(drugs_to_labels_path, 'r') as f:
        for line in f:
            folder, drug, label = line.split()
            drug_to_labels_dict[drug] = int(label)
            folder_to_label[folder] = int(label)
            folder_to_drug[folder] = drug

    return drug_to_labels_dict, folder_to_label, folder_to_drug

def normalize_and_mask(img, eps=1e-6, mask_threshold=0.1):
    """
    Remove morphology-related brightness (TMRM) from MitoTracker
    to emphasize functional membrane potential signal.

    Args:
        img: Tensor [B, C, Z, H, W]
             channel 0 = TMRM (morphology)
             channel 1 = MitoTracker (function + morphology)
        eps: small value to prevent divide-by-zero
        mask_threshold: relative threshold for mitochondrial mask
    """
    tmrm = img[0, :, :]      # morphology only
    mitotk = img[1, :, :]    # morphology + function

    # --- Step 1: background subtraction ---
    tmrm = tmrm - tmrm.min()
    mitotk = mitotk - mitotk.min()

    # --- Step 2: normalize both to [0, 1] ---
    tmrm = tmrm / (tmrm.max() + eps)
    mitotk = mitotk / (mitotk.max() + eps)

    # --- Step 3: remove morphology contribution ---
    # Divide MitoTracker by TMRM to isolate functional signal
    functional = mitotk / (tmrm + eps)
    functional = functional / (functional.max() + eps)

    # --- Step 4: create mitochondrial mask ---
    mask = (tmrm > mask_threshold * tmrm.max()).astype(np.float32)

    # --- Step 5: apply mask ---
    functional_masked = functional * mask

    return functional_masked, mask



def view_4d_image_with_sliders(image_filepath, position = None):
    """
    Views a 4D (time, z, y, x, channels) image from a .npy file with sliders
    for time and z-slice selection.

    Args:
        image_filepath (str): Path to the .npy file containing the 4D image data.
    """
    try:
        image_data = np.load(image_filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {image_filepath}")
        return
    except Exception as e:
        print(f"Error loading image data: {e}")
        return
    

    # Assuming data shape is (time_points, z_slices, y_dim, x_dim, channels)
    # Adjust this if your data has a different channel position
    if image_data.ndim != 5:
        print(f"Error: Expected 5 dimensions (time, z, y, x, channels), but got {image_data.ndim}")
        print("Please ensure your .npy file has the correct shape.")
        return

    time_points,num_channels, z_slices, y_dim, x_dim = image_data.shape

    if num_channels != 2:
        print(f"Warning: Expected 2 channels, but found {num_channels}. Displaying the first two if available.")

    # Initial display parameters
    initial_time = 0
    initial_z = 0

    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) # One subplot for each channel
    plt.subplots_adjust(left=0.1, bottom=0.25)
    folder , basename = os.path.split(image_filepath)
    
    folder = os.path.basename(folder)
    _, folder_to_label, folder_to_drug = load_folder_label_maps()

    drug = folder_to_drug[folder]
    # Load folder_to_label and label_names from the drugs_to_labels.txt mapping file,
    # Display initial images for both channels
    im1 = ax[0].imshow(image_data[initial_time,0, initial_z, :, :], cmap='gray')
    ax[0].set_title(f"Channel 1 (Time: {initial_time}, Z: {initial_z})")
    ax[0].axis('off')

    im2 = ax[1].imshow(image_data[initial_time,1, initial_z, :, :], cmap='gray')
    ax[1].set_title(f"Channel 2 (Time: {initial_time}, Z: {initial_z})")
    ax[1].axis('off')
    if position is None:

        plt.suptitle(f"Folder: {folder}, Basename: {basename}")
    else:
        plt.suptitle(f"Folder: {folder}, Basename: {basename}, Position: {position}, Drug: {drug}")

    # # im3 = ax[2].imshow(image_data[initial_time,1, initial_z, :, :]-image_data[initial_time,0, initial_z, :, :], cmap='gray')
    # im3 = ax[2].imshow(normalize_and_mask(image_data[initial_time,:, initial_z, :, :])[0], cmap = 'gray')
    # ax[2].set_title(f"Channel 2-1 (Time: {initial_time}, Z: {initial_z})")
    # ax[2].axis('off')

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
        ax[0].set_title(f"Channel 1 (Time: {current_time}, Z: {current_z})")

        if num_channels >= 2:
            im2.set_data(image_data[current_time, 1,current_z, :, :])
            ax[1].set_title(f"Channel 2 (Time: {current_time}, Z: {current_z})")
            # im3.set_data(image_data[current_time, 1,current_z, :, :]-image_data[current_time, 0,current_z, :, :])
            # im3.set_data(normalize_and_mask(image_data[current_time, :,current_z, :, :])[0])
            # ax[2].set_title(f"Channel 2-1 (Time: {current_time}, Z: {current_z})")
        else:
            ax[1].set_title(f"Channel 2 (Not available)")
            im2.set_data(np.zeros_like(image_data[current_time, 0,current_z, :, :])) # Show black if no second channel
            # ax[2].set_title(f"Channel 2-1 (Not available)")
            # im3.set_data(np.zeros_like(image_data[current_time, 0,current_z, :, :])) # Show black if no second channel
            


        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    z_slider.on_changed(update)

    # --- Add Save Movie Button ---
    # ax_save = plt.axes([0.8, 0.9, 0.15, 0.05])
    # btn_save = Button(ax_save, 'Save Movie', color='lightblue', hovercolor='skyblue')

    def save_movie(event):
        print("Preparing movie export...")

        mode = input("Animate over [time/z]? ").strip().lower()
        if mode not in ["time", "z"]:
            print("Invalid choice. Please type 'time' or 'z'.")
            return

        save_path = input("Enter output filename (e.g. movie.mp4): ").strip()
        fps = 5  # frames per second

        frames = range(time_points) if mode == "time" else range(z_slices)

        def animate(i):
            if mode == "time":
                t, z = i, int(z_slider.val)
            else:
                t, z = int(time_slider.val), i

            im1.set_data(image_data[t, 0, z, :, :])
            im2.set_data(image_data[t, 1, z, :, :])
            # im3.set_data(normalize_and_mask(image_data[t, :, z, :, :])[0])

            ax[0].set_title(f"Ch1 (t={t}, z={z})")
            ax[1].set_title(f"Ch2 (t={t}, z={z})")
            # ax[2].set_title(f"Ch2-1 (t={t}, z={z})")
            return [im1, im2] # im3

        ani = animation.FuncAnimation(fig, animate, frames=frames, blit=False, repeat=False)

        try:
            ani.save(save_path, fps=fps, writer='ffmpeg')
            print(f"✅ Movie saved successfully: {save_path}")
        except Exception as e:
            print(f"❌ Error saving movie: {e}")
            print("Make sure FFmpeg is installed (e.g., `sudo apt install ffmpeg`).")

    # btn_save.on_clicked(save_movie)
    
    plt.show()

if __name__ == "__main__":
    # --- Create a dummy .npy file for testing ---
    # This simulates a 4D image with 2 channels
    print("Creating a dummy 4D image file 'dummy_4d_image.npy' for demonstration...")
    # Example dimensions: (5 time points, 10 z-slices, 64 y, 64 x, 2 channels)
    # dummy_time_points = 5
    # dummy_z_slices = 10
    # dummy_y_dim = 64
    # dummy_x_dim = 64
    # dummy_channels = 2

    # dummy_image_data = np.zeros((dummy_time_points, dummy_z_slices, dummy_y_dim, dummy_x_dim, dummy_channels), dtype=np.uint8)

    # for t in range(dummy_time_points):
    #     for z in range(dummy_z_slices):
    #         # Create some visual variation for Channel 1 (e.g., a gradient based on time and z)
    #         dummy_image_data[t, z, :, :, 0] = (np.sin(np.linspace(0, np.pi * (t + z) / 20, dummy_y_dim)).reshape(-1, 1) *
    #                                           np.cos(np.linspace(0, np.pi * (t + z) / 15, dummy_x_dim))).astype(np.float32) * 128 + 127

    #         # Create some visual variation for Channel 2 (e.g., a different pattern)
    #         dummy_image_data[t, z, :, :, 1] = (np.sin(np.linspace(0, np.pi * (t + 1) / 10, dummy_y_dim)).reshape(-1, 1) +
    #                                           np.cos(np.linspace(0, np.pi * (z + 1) / 8, dummy_x_dim))).astype(np.float32) * 80 + 100

    # # Ensure data is in a displayable range if not already
    # dummy_image_data = (dummy_image_data - dummy_image_data.min()) / (dummy_image_data.max() - dummy_image_data.min()) * 255
    # dummy_image_data = dummy_image_data.astype(np.uint8)


    # dummy_filename = "dummy_4d_image.npy"
    # np.save(dummy_filename, dummy_image_data)
    # print(f"Dummy file '{dummy_filename}' created successfully.")

    # --- Run the viewer with the dummy file ---
    your_image_path = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240815/000079.npy"
    view_4d_image_with_sliders(your_image_path)
    # view_4d_image_with_sliders(dummy_filename)