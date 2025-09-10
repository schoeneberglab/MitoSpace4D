import os.path as osp
import imageio
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_video(vid_path):
    try:
        vid = imageio.get_reader(vid_path, format='ffmpeg')
        writer = imageio.get_writer(vid_path.replace("mtg_", "combined_"), fps=10, codec='libx264')

        for frame in vid:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            combined_frame = np.zeros((*gray_frame.shape, 3), dtype=np.uint8)
            combined_frame[..., 1] = np.clip(gray_frame / 1.3, 0, 255).astype(np.uint8)
            combined_frame[..., 0] = gray_frame
            combined_frame[..., 2] = gray_frame

            # increase the contrast of the combined frame
            combined_frame = cv2.convertScaleAbs(combined_frame, alpha=1.5, beta=0)
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

            writer.append_data(combined_frame)

        writer.close()
        vid.close()
        return True  # success
    except Exception as e:
        print(f"Error processing {vid_path}: {e}")
        return False  # failure


if __name__ == '__main__':
    source_dir = '/u/earkfeld/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/videos'
    video_ids = [str(i).zfill(6) for i in range(13000)]

    mito_vids_path = [osp.join(source_dir, f"mtg_{vid_id}.mp4") for vid_id in video_ids]
    tmrm_vids_path = [osp.join(source_dir, f"tmrm_{vid_id}.mp4") for vid_id in video_ids]

    # process mito videos
    process_video(mito_vids_path[0])

