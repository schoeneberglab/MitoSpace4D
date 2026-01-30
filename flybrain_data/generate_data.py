import nd2
import numpy as np
import napari
from skimage.transform import resize
# Read the file into a NumPy array
import tqdm
# my_array = nd2.imread("mdi-GFP-60x_B001 - Deconvolved.nd2")

# # You can then check its shape, dtype, etc.
# print(f"Image shape: {my_array.shape}")
# print(f"Image data type: {my_array.dtype}")
# my_array = nd2.imread("Tom20_A_B001_60X - Deconvolved.nd2")

# # You can then check its shape, dtype, etc.
# print(f"Image shape: {my_array.shape}")
# print(f"Image data type: {my_array.dtype}")

def preprocess_nd2(file_path, target_size=(256, 256)):
    with nd2.ND2File(file_path) as s:
        # Load as a dask array or numpy array
        # Shape: (Frames, Channels, Y, X)
        img_data = s.asarray() 
        
    num_frames, num_channels = img_data.shape[0], img_data.shape[1]
    
    # Initialize output array (float32 is safer for resized data)
    resized_images = np.zeros((num_frames, num_channels, *target_size), dtype=np.float32)

    for f in range(num_frames):
        for c in range(num_channels):
            # Resize each channel independently
            # preserve_range=True keeps the uint16 scale (0-65535)
            resized_images[f, c] = resize(img_data[f, c], target_size, 
                                          anti_aliasing=True, 
                                          preserve_range=True)
            
    return resized_images

def divide_into_slices(image_path, slice_size=(128, 128)):
    with nd2.ND2File(image_path) as s:
        img_data = s.asarray()
    num_frames, num_channels = img_data.shape[0], img_data.shape[1]

    # Divide each frame/channel into non-overlapping chunks of slice_size using a sliding window (convolutional approach)
    slices = []
    stride_y, stride_x = slice_size
    
    # Compute number of slices in each dimension
    n_y = img_data.shape[2] // stride_y
    n_x = img_data.shape[3] // stride_x
    # For each (i, j), collect all frames and all channels for that chunk location,
    # combine them into a single array with shape (1, all_f, c, 256, 256)
    for i in tqdm.tqdm(range(n_y)):
        for j in tqdm.tqdm(range(n_x)):
            # Gather the chunk (256x256) from all frames and all channels at (i, j)
            
            # for f_idx in range(num_frames):
            #     frame_chunks = []
            #     for c_idx in range(num_channels):
            chunk = img_data[:, 0:1,
                            i*stride_y : (i+1)*stride_y,
                            j*stride_x : (j+1)*stride_x]
            chunk = np.repeat(chunk, num_channels, axis=1)

            chunk = np.expand_dims(chunk, axis=0)
            chunk = chunk.astype(np.float32)
            # chunk = np.transpose(chunk, (0, 2,1,3,4))
            # If slice_size < 256, resize chunk spatially to (256, 256) using interpolation
            if slice_size[0] < 256 or slice_size[1] < 256:
                # chunk shape: (1, all_f, c, y, x)
                target_shape = (256, 256)
                new_chunk = np.zeros((chunk.shape[0], chunk.shape[1], chunk.shape[2], target_shape[0], target_shape[1]), dtype=np.float32)
                for batch in range(chunk.shape[0]):
                    for z in range(chunk.shape[1]):
                        for chan in range(chunk.shape[2]):
                            new_chunk[batch, z, chan] = resize(
                                chunk[batch, z, chan],
                                target_shape,
                                anti_aliasing=True,
                                preserve_range=True
                            )
                chunk = new_chunk

            # For each chunk, split across z (frame) slices into 2 halves, pad to 60 in both halves, center original z
            # chunk: (1, all_f, c, 256, 256) -> we want (1, 60, c, 256, 256) two times, pad/crop along axis=1
            all_f = chunk.shape[1]
            z_target = 60
            # First half indices and second half indices (across z)
            mid = all_f // 2  # e.g. 61//2 = 30
            idx1 = slice(0, mid)
            idx2 = slice(mid, all_f)
            # chunk1 = None
            # chunk2 = None
            
            # chunk1 = chunk[:, idx1, ...]
            # chunk2 = chunk[:, idx2, ...]    
            for idx in [idx1, idx2]:
                z_part = chunk[:, idx, ...]  # shape (1, ?, c, 256, 256)
                z_len = z_part.shape[1]
                # Pad to 60 (centered)
                pad_left = (z_target - z_len) // 2
                pad_right = z_target - z_len - pad_left
                if pad_left > 0 and pad_right > 0:
                    pad_width = [(0,0), (pad_left, pad_right), (0,0), (0,0), (0,0)]
                    z_padded = np.pad(z_part, pad_width, mode='constant')
                    
                else:
                    z_padded = z_part
                z_padded = np.transpose(z_padded, (0, 2,1,3,4))
                slices.append(z_padded)
            
            
            #         frame_chunks.append(chunk)
            #     # frame_chunks: list with length == num_channels, each item shape (256,256)
            #     frame_chunks = np.stack(frame_chunks, axis=0)  # (c, 256, 256)
            #     chunks.append(frame_chunks)
            # # chunks: list with length == num_frames, each item (c, 256, 256)
            # combined_chunk = np.stack(chunks, axis=0)  # (all_f, c, 256, 256)
            # combined_chunk = combined_chunk[np.newaxis, ...]  # (1, all_f, c, 256, 256)
            # slices.append(chunk1)
            # slices.append(chunk2)
            # slices.append(chunk)

    return slices

file_paths = ["original/mdi-GFP-60x_B001 - Deconvolved.nd2", "original/Tom20_A_B001_60X - Deconvolved.nd2", "original/VAChT-myc-GE_B001-R - Deconvolved.nd2"]

for file_path in file_paths:
    slices = divide_into_slices(file_path)
    print(f"Number of slices: {len(slices)}")
    print(f"Slice shape: {slices[0].shape}")
    print(f"Slice data type: {slices[0].dtype}")

    output_path = f"chunks_128/{file_path.split('/')[-1].split(' - ')[0]}-slices.npy"   
    np.save(output_path, slices)

# file_path = "mdi-GFP-60x_B001 - Deconvolved.nd2"
# resized_images = preprocess_nd2(file_path)
# print(f"Resized image shape: {resized_images.shape}")
# print(f"Resized image data type: {resized_images.dtype}")

# output_path = "mdi-GFP-60x_B001-resized.npy"
# np.save(output_path, resized_images)

# file_path = "Tom20_A_B001_60X - Deconvolved.nd2"
# viewer = napari.Viewer()
# viewer.open(file_path)
# napari.run() # Keep the viewer open
