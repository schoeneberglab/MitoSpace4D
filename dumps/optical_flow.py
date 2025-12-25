import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def invert_bg(X, black_channel=None):
    white_bg = np.ones_like(X)
    white_bg[:, :, [0, 1]] -= np.repeat(X[:, :, None, 2], 2, -1)
    white_bg[:, :, [0, 2]] -= np.repeat(X[:, :, None, 1], 2, -1)
    white_bg[:, :, [1, 2]] -= np.repeat(X[:, :, None, 0], 2, -1)

    if black_channel is not None:
        for h in range(black_channel.shape[0]):
            for w in range(black_channel.shape[1]):
                if black_channel[h][w] == 1.:
                    white_bg[h, w, :] = 0.

    else:
        for h in range(white_bg.shape[0]):
            for w in range(white_bg.shape[1]):
                if X[h, w, 2] == 1.:
                    white_bg[h, w, :] = 0.

    white_bg = np.clip(white_bg, 0., 1.)
    return white_bg

def compute_optical_flow_arrows(frame1, frame2, step=10, disp_frame=None, crop_params=None):
    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Create a grid of points for the arrows
    h, w = frame1.shape
    y, x = np.mgrid[step//2:h:step, step//2:w:step]  # Sample points for arrows
    fx, fy = flow[y, x].T  # Get flow vectors at sampled points

    # Plot arrows on the frame
    plt.figure(figsize=(8, 8))
    if disp_frame is None:
        plt.imshow(frame1, cmap='gray')
    plt.imshow(disp_frame, cmap='gray')  # Show original frame
    plt.quiver(x, y, fx, fy, angles="xy", scale_units="xy", scale=1.5, color="blue")
    plt.axis("off")

    plt.show()


def compute_optical_flow(frame1, frame2):
    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow to HSV for visualization
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255  # Set saturation to max

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Compute magnitude and angle
    hsv[..., 0] = ang * 180 / np.pi / 2  # Convert angle to HSV color
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Convert to RGB

    return flow_rgb

def get_foreground(img):
    # Apply a simple threshold to get the foreground mask
    _, fg_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return fg_mask


def crop(img, bbox):
    return img[bbox[0]:bbox[1], bbox[2]:bbox[3]]

if __name__ == '__main__':
    # imgs_path = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240802/000586.npy' # tbhp slow
    # imgs_path = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240802/000395.npy' # tbhp fast
    imgs_path = '/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240809/000364.npy' # oligo
    # imgs_path = "/media/dhruvagarwal/easystore/MitoSpace4D/data/2024_data/processed_data/20240805/000561.npy" # h2o2
    imgs = np.load(imgs_path) # (t, c, z, h, w)
    imgs = imgs[:, 0, :, :, :]  # MTG channel
    imgs_mip = np.max(imgs, axis=1)  # MIP along time

    # imgs_mip = np.clip(imgs_mip, 0, 10000) / 10000  # Normalize to [0, 1]
    # imgs_mip = (imgs_mip * 255).astype(np.uint8)  # Convert to 8-bit

    # get the foreground mask
    masks = [get_foreground(img) for img in imgs_mip]


    # flow_images = [compute_optical_flow(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    #
    # # Plot the first few optical flow results
    # fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Show first 5 flows
    # for i, ax in enumerate(axes):
    #     ax.imshow(flow_images[i])
    #     ax.set_title(f"Flow: Frame {i} → {i + 1}")
    #     ax.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    # Compute frame differences
    # diffs = [cv2.absdiff(masks[i], masks[i + 1]) for i in range(len(masks) - 1)]
    #
    # # Plot the first few frame differences
    # fig, axes = plt.subplots(1, 5, figsize=(15, 5))  # Show first 5 differences
    # for i, ax in enumerate(axes):
    #     ax.imshow(diffs[i], cmap='hot')  # Use heatmap for better visualization
    #     ax.set_title(f"Δ Frame {i} → {i + 1}")
    #     ax.axis("off")
    #
    # plt.tight_layout()
    # plt.show()

    # Compute optical flow arrows
    imgs_mip = np.clip(imgs_mip, 0, 10000) / 10000  # Normalize to [0, 1]
    disp_frame = np.zeros((imgs_mip.shape[1], imgs_mip.shape[2], 3))
    disp_frame[:, :, 1] = imgs_mip[0]
    disp_frame = invert_bg(disp_frame)
    compute_optical_flow_arrows(masks[0], masks[-1], step=5, disp_frame=disp_frame, crop_params=(50, 200, 60, 200))


