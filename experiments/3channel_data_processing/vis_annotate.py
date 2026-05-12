import csv
import os

import matplotlib.pyplot as plt
import numpy as np

data_dir = "/mnt/aquila/ssd_processing/Others/MitoSpace4D/andre_3color_cancer/20260129-0/"
out_csv = os.path.join(os.getcwd(), "dead_cell_paths.csv")

subdirs = sorted(
    os.path.join(data_dir, d)
    for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
)
print(subdirs)

dead_cell_paths = []

fig, ax = plt.subplots()
pressed = {"key": None}


def on_key(event):
    if event.key in ("d", "l"):
        pressed["key"] = event.key


fig.canvas.mpl_connect("key_press_event", on_key)

plt.ion()
plt.show(block=False)

total = len(subdirs)
for i, subdir in enumerate(subdirs):
    files = sorted(f for f in os.listdir(subdir) if f.endswith(".npy"))
    if not files:
        continue

    file_paths = [os.path.abspath(os.path.join(subdir, f)) for f in files]
    last = np.load(file_paths[-1])
    # maximum intensity projecton
    mip = np.max(last, axis=0)

    remaining = total - i - 1
    ax.clear()
    ax.imshow(mip, cmap="gray")
    ax.set_title(
        f"[{i + 1}/{total}] {os.path.basename(subdir)}  —  "
        f"{remaining} left  —  dead={len(dead_cell_paths)} paths  —  d = dead, l = skip"
    )
    ax.set_axis_off()

    pressed["key"] = None
    while pressed["key"] is None and plt.fignum_exists(fig.number):
        plt.pause(0.1)

    if not plt.fignum_exists(fig.number):
        break

    if pressed["key"] == "d":
        dead_cell_paths.extend(file_paths)

plt.close(fig)

with open(out_csv, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["path"])
    for p in dead_cell_paths:
        writer.writerow([p])

print(f"Wrote {len(dead_cell_paths)} paths to {out_csv}")