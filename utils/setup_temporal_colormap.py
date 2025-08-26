import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt

# def setup_image_times_colormap(embeddings_dir, cmap="viridis"):
#     infile = osp.join(embeddings_dir, 'image_times.npy')
#     image_times = np.load(infile)

#     # Set up a colormap for the image times
#     cmap = plt.cm.get_cmap(cmap)
#     image_times_colors = np.array([cmap(i / image_times.max())[:3] for i in image_times])
#     image_times_colors = (image_times_colors * 255).astype(np.uint8)

#     # Save the colormap
#     outfile = osp.join(embeddings_dir, 'image_times_colormap.npy')
#     np.save(outfile, image_times_colors)
#     print(f"Saved image times colormap to: {outfile}")


def create_temporal_colormap(image_path_file, cell_to_region_file, outdir="", cmap="viridis", n_frames=20, single_frames=True):
    """ Sets up temporal color map and saves to: <outdir>/temporal_colormap.npy """
    df_samples = pd.read_csv(image_path_file, dtype=str, header=None, delimiter=" ")
    df_regions = pd.read_csv(cell_to_region_file)

    # Set the header
    df_samples.columns = ["fpath"]
    df_samples['condition'] = df_samples['fpath'].apply(lambda x: x.split("/")[-2])
    df_samples['filename'] = df_samples['fpath'].apply(lambda x: x.split("/")[-1].split(".")[0])
    df_samples['cell_id'] = df_samples['filename'].apply(lambda x: int(x.split("-")[0]))
    df_samples['cell_tid'] = df_samples['filename'].apply(lambda x: int(x.split("-")[1]))
    df_samples['region_id'] = -1 # Set up an empty column for the region id

    for condition in df_regions["Data Path"].unique():
        df_regions_condition = df_regions[df_regions["Data Path"] == condition]
        df_samples_condition = df_samples[df_samples["condition"] == condition]

        for i, row in df_regions_condition.iterrows():
            cell_id_start = row["Cell ID Start"]
            try:
                cell_id_end = df_regions_condition.loc[i + 1, "Cell ID Start"]
            except:
                # Set infinite integer value
                cell_id_end = np.inf

            current_region_id = row["Region ID"]
            for i, sample_row in df_samples_condition.iterrows():
                current_cell_id = sample_row['cell_id']

                if current_cell_id >= cell_id_start and current_cell_id < cell_id_end:
                    df_samples_condition.at[i, "region_id"] = current_region_id

        df_samples.update(df_samples_condition)

    # Compute time ids (assuming no delay between regions)
    time_id_fn = lambda region_id, cell_tid: 3*region_id + cell_tid
    single_frame_time_id_fn = lambda region_id, cell_tid, frame_id, n_frames: (3*region_id*n_frames) + (cell_tid * n_frames) + frame_id
    df_samples["time_id"] = df_samples.apply(lambda x: time_id_fn(x["region_id"], x["cell_tid"]), axis=1)
    
    # Set up a plasma color map as a function of time_id (rgb, 0-255)
    # df_samples['color'] = df_samples['time_id'].apply(lambda x: (np.array(plt.cm.plasma(x / df_samples['time_id'].max())[:3]) * 255).astype(int))
    
    # Get the color map from matplotlib
    cmap = plt.cm.get_cmap(cmap)

    if single_frames:
        single_frame_times = []
        # Iterate over all rows in the dataframe
        for i, row in df_samples.iterrows():
            for frame_id in range(n_frames):
                single_frame_times.append(single_frame_time_id_fn(row["region_id"], row["cell_tid"], frame_id, n_frames))
        
        # Set up the color map for each value in single frames
        colors = np.array([cmap(i / max(single_frame_times))[:3] for i in single_frame_times])
        print(np.max(single_frame_times))
        print(np.min(single_frame_times))

        # Save the colormap
        np.save(osp.join(outdir, f'frame_temporal_colormap.npy'), colors)

    else:
        # Process each movie as a whole
        df_samples['temporal_colors'] = df_samples['time_id'].apply(lambda x: (np.array(cmap(x / df_samples['time_id'].max())[:3])))
        df_samples['region_colors'] = df_samples['region_id'].apply(lambda x: (np.array(cmap(x / df_samples['region_id'].max())[:3])))
        temporal_colors_arr = np.vstack(df_samples['temporal_colors'].values)
        region_colors_arr = np.vstack(df_samples['region_colors'].values)

        temporal_outfile = osp.join(outdir, 'temporal_colormap.npy')
        np.save(temporal_outfile, temporal_colors_arr)
        print(f"Saved color map to: {temporal_outfile}")

        region_outfile = osp.join(outdir, 'region_colormap.npy')
        np.save(region_outfile, region_colors_arr)
        print(f"Saved color map to: {region_outfile}")

        # Display viridis color bar (as a reference)
        plt.figure(figsize=(6, 1))
        plt.title("Time")
        plt.imshow(np.linspace(0, 1, 256)[np.newaxis, ...], aspect="auto", cmap="viridis")
        plt.axis("off")
        plt.show()

def create_colormap(img_path_file, cell_region_map, outdir="", cmap="viridis", n_frames=20, single_frames=True):
    """ Sets up a color map by region id. """

    # df_region = pd.read_csv(cell_region_map)
    # img_paths = np.loadtxt(img_path_file, dtype=str).tolist()

    # """
    # Example image path:
    # /run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer/20250722-2/000001-0.npy
    # """

    # region_ids = []
    # for img_path in img_paths:
    #     condition_id = img_path.split("/")[-2]
    #     sample_id = img_paths.split("/")[-1].split(".")[0]

    df_samples = pd.read_csv(img_path_file, dtype=str, header=None, delimiter=" ")
    df_regions = pd.read_csv(cell_region_map)

    # Set the header
    df_samples.columns = ["fpath"]
    df_samples['condition'] = df_samples['fpath'].apply(lambda x: x.split("/")[-2])
    df_samples['filename'] = df_samples['fpath'].apply(lambda x: x.split("/")[-1].split(".")[0])
    df_samples['cell_id'] = df_samples['filename'].apply(lambda x: int(x.split("-")[0]))
    df_samples['cell_tid'] = df_samples['filename'].apply(lambda x: int(x.split("-")[1]))
    df_samples['region_id'] = -1 # Set up an empty column for the region id

    for condition in df_regions["Data Path"].unique():
        df_regions_condition = df_regions[df_regions["Data Path"] == condition]
        df_samples_condition = df_samples[df_samples["condition"] == condition]

        for i, row in df_regions_condition.iterrows():
            cell_id_start = row["Cell ID Start"]
            try:
                cell_id_end = df_regions_condition.loc[i + 1, "Cell ID Start"]
            except:
                # Set infinite integer value
                cell_id_end = np.inf

            current_region_id = row["Region ID"]
            for i, sample_row in df_samples_condition.iterrows():
                current_cell_id = sample_row['cell_id']

                if current_cell_id >= cell_id_start and current_cell_id < cell_id_end:
                    df_samples_condition.at[i, "region_id"] = current_region_id

        df_samples.update(df_samples_condition)

    # time = region offset + cell instance offset + frame_id
    single_frame_time_id_fn = lambda region_id, cell_tid, frame_id, n_frames: (3*region_id*n_frames) + (cell_tid * n_frames) + frame_id
    max_time = single_frame_time_id_fn(df_samples['region_id'].max(), df_samples['cell_tid'].max(), n_frames-1, n_frames)
    min_time = 0

    # Set up the colormap
    cmap = plt.get_cmap(cmap)

    region_norm = plt.Normalize(vmin=df_samples['region_id'].min(), vmax=df_samples['region_id'].max())
    temporal_norm = plt.Normalize(vmin=min_time, vmax=max_time)

    region_colors = []
    temporal_colors = []

    from tqdm import tqdm
    for i, row in tqdm(df_samples.iterrows(), total=df_samples.shape[0]):
        if single_frames:
            region_colors.extend([cmap(region_norm(row['region_id']))[:3]] * n_frames)
            temporal_colors.extend([cmap(temporal_norm(single_frame_time_id_fn(row['region_id'], row['cell_tid'], frame_id, n_frames)))[:3] for frame_id in range(n_frames)])
        else:
            region_colors.append(cmap(region_norm(row['region_id']))[:3])
            temporal_colors.append(cmap(temporal_norm(single_frame_time_id_fn(row['region_id'], row['cell_tid'], 0, n_frames)))[:3])
    
    # Save the colormaps
    np.save("region_colors.npy", region_colors)
    np.save("temporal_colors.npy", temporal_colors)
    print("Colormaps saved.")

if __name__ == "__main__":
    cell_region_map = "/run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer/cell_to_region.csv"
    
    # img_filepaths = "/home/earkfeld/Projects/MitoSpace4D/runs/embeddings_full/image_paths.csv"
    # img_filepaths = "/home/earkfeld/Projects/MitoSpace4D/runs/embeddings_test/image_paths.csv"
    img_filepaths = "/mnt/DATA_01/Eric/mitospace4d_data/runs/embeddings_test/image_paths.csv"

    # img_filepaths = "/home/earkfeld/Projects/MitoSpace4D/runs/embeddings_cancer/image_paths.csv"
    # cell_region_map = "/run/user/1002/gvfs/smb-share:server=aquila0.jslab.ucsd.edu,share=ssd_processing/Others/MitoSpace4D/2025_summer/cell_to_region.csv"
    
    create_colormap(img_filepaths, cell_region_map)
    # create_region_colormap(img_filepaths, cell_region_map)