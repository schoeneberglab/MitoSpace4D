import os.path as osp
import imageio
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_video(args):
    mito_vid_path, tmrm_vid_path, combined_vid_path = args

    try:
        mito_vid = imageio.get_reader(mito_vid_path, format='ffmpeg')
        tmrm_vid = imageio.get_reader(tmrm_vid_path, format='ffmpeg')
        writer = imageio.get_writer(combined_vid_path, fps=10, codec='libx264')

        for mito_frame, tmrm_frame in zip(mito_vid, tmrm_vid):
            mito_gray = cv2.cvtColor(mito_frame, cv2.COLOR_RGB2GRAY)
            tmrm_gray = cv2.cvtColor(tmrm_frame, cv2.COLOR_RGB2GRAY)

            combined_frame = np.zeros((*mito_gray.shape, 3), dtype=np.uint8)
            combined_frame[..., 1] = np.clip(mito_gray / 1.3, 0, 255).astype(np.uint8)
            combined_frame[..., 0] = tmrm_gray
            combined_frame[..., 2] = tmrm_gray

            # increase the contrast of the combined frame
            combined_frame = cv2.convertScaleAbs(combined_frame, alpha=1.5, beta=0)
            combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

            writer.append_data(combined_frame)

        writer.close()
        mito_vid.close()
        tmrm_vid.close()
        return True  # success
    except Exception as e:
        print(f"Error processing {combined_vid_path}: {e}")
        return False  # failure

if __name__ == '__main__':
    source_dir = '/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/videos'
    video_ids = [str(i).zfill(6) for i in range(13000)]

    mito_vids_path = [osp.join(source_dir, f"mtg_{vid_id}.mp4") for vid_id in video_ids]
    tmrm_vids_path = [osp.join(source_dir, f"tmrm_{vid_id}.mp4") for vid_id in video_ids]
    combined_vids_path = [osp.join(source_dir, f"combined_{vid_id}.mp4") for vid_id in video_ids]

    all_args = list(zip(mito_vids_path, tmrm_vids_path, combined_vids_path))

    num_workers = multiprocessing.cpu_count()  # Use all available cores
    # num_workers = 1
    print(f"Using {num_workers} workers.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_video, all_args), total=len(all_args)))

    num_success = sum(results)
    print(f"Done! {num_success}/{len(all_args)} videos processed successfully.")
