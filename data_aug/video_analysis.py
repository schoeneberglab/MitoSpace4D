# Python analysis visible to user: analyze two videos and produce plots + representative frames.
# This will run in the notebook and display results.
import cv2, os, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
import pandas as pd

video_paths = ["/mnt/data/mv1.avi", "/mnt/data/mv2.avi"]
labels = ["Video 1", "Video 2"]

def analyze_video(path, sample_stride=2, max_frames=500):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    results = []
    frames = []
    prev_gray = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_stride != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.copy())
        # basic denoise
        g = cv2.medianBlur(gray, 3)
        # intensity metric
        mean_int = float(np.mean(g))
        # threshold for segmentation (Otsu)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        # connected components via findContours (assume bright structures)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c)>5]
        # shape metrics (approximate length as max dimension of bounding box)
        bboxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>5]
        aspects = []
        for (x,y,w,h) in bboxes:
            if h>0:
                aspects.append(float(w)/h if h>0 else 0)
        mean_area = float(np.mean(areas)) if areas else 0.0
        median_area = float(np.median(areas)) if areas else 0.0
        count = len(areas)
        mean_aspect = float(np.mean(aspects)) if aspects else 0.0
        # optical flow (motion magnitude) between this and prev frame
        motion = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion = float(np.mean(mag))
        prev_gray = g.copy()
        results.append({
            "frame_idx": idx,
            "mean_intensity": mean_int,
            "contour_count": count,
            "mean_area": mean_area,
            "median_area": median_area,
            "mean_aspect": mean_aspect,
            "motion": motion,
            "time_s": idx / fps if fps>0 else idx
        })
        idx += 1
        if len(results) >= max_frames:
            break
    cap.release()
    df = pd.DataFrame(results)
    return df, frames, fps, frame_count

analyses = []
for p in video_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found")
    df, frames, fps, frame_count = analyze_video(p, sample_stride=2, max_frames=800)
    analyses.append((df, frames, fps, frame_count))

# Display summary plots for both videos
plt.figure(figsize=(10,6))
for i,(df,_,fps,fc) in enumerate(analyses):
    plt.plot(df['time_s'], df['contour_count'], label=f"{labels[i]} contours")
plt.xlabel("Time (s)")
plt.ylabel("Object count (segmented mitochondria)")
plt.legend()
plt.title("Segmented object count over time")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i,(df,_,fps,fc) in enumerate(analyses):
    plt.plot(df['time_s'], df['mean_area'], label=f"{labels[i]} mean area")
plt.xlabel("Time (s)")
plt.ylabel("Mean segmented area (pixels)")
plt.legend()
plt.title("Mean object area over time")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i,(df,_,fps,fc) in enumerate(analyses):
    plt.plot(df['time_s'], df['mean_intensity'], label=f"{labels[i]} mean intensity")
plt.xlabel("Time (s)")
plt.ylabel("Mean frame intensity (0-255)")
plt.legend()
plt.title("Mean frame intensity over time")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
for i,(df,_,fps,fc) in enumerate(analyses):
    plt.plot(df['time_s'], df['motion'], label=f"{labels[i]} motion mag")
plt.xlabel("Time (s)")
plt.ylabel("Mean optical flow magnitude")
plt.legend()
plt.title("Average motion magnitude over time")
plt.tight_layout()
plt.show()

# Create montage of representative frames (start, middle, end) for each video
def montage_from_frames(frames, name):
    if not frames:
        return None
    n = len(frames)
    picks = [0, n//2, n-1]
    imgs = [frames[i] for i in picks]
    h,w = imgs[0].shape
    mont = np.concatenate([cv2.cvtColor(cv2.resize(img,(w,h)), cv2.COLOR_GRAY2BGR) for img in imgs], axis=1)
    return mont

for i,(_,frames,fps,fc) in enumerate(analyses):
    mont = montage_from_frames(frames, labels[i])
    if mont is not None:
        plt.figure(figsize=(8,3))
        plt.imshow(mont[:,:,::-1])
        plt.axis('off')
        plt.title(f"{labels[i]} — representative frames (start | middle | end)")
        plt.show()

# Display small data summaries as a table
summary_rows = []
for i,(df,_,fps,fc) in enumerate(analyses):
    summary_rows.append({
        "Video": labels[i],
        "Frames analyzed": len(df),
        "Approx total frames": fc,
        "FPS": fps,
        "Mean intensity (overall)": float(df['mean_intensity'].mean()),
        "Mean object count": float(df['contour_count'].mean()),
        "Mean motion": float(df['motion'].mean()),
        "Mean area": float(df['mean_area'].mean())
    })
summary_df = pd.DataFrame(summary_rows)
import caas_jupyter_tools as cjt
cjt.display_dataframe_to_user("Video analysis summary", summary_df)
