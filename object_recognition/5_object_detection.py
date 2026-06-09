import h5py
import numpy as np
import cv2
import tifffile
from image_data_dvp import ImageManager
from find_h5_dataset import get_h5_datasets
import time

# ================== CONFIGURATION ==================
filename = "C:\\Users\\Yoginder Singh\\Downloads\\251204_150955.h5"
dataset_index = 0                      # which dataset inside HDF5
frame_start = 0                        # optional: first frame to process
frame_end = None                       # optional: last frame (None = all)

# Detection parameters (tuned for your data)
detection_params = {
    "channel": 0,
    "bitdepth": 12,
    "norm_factor": 16,
    "intensity_threshold": 10,
    "nms_d_min": 7,
    "R_cluster": 100,
    "N_cluster": 20
}

# Output file
output_tiff = "annotated_stack.tif"
# ===================================================

# ---- Load data ----
data = get_h5_datasets(filename, dataset_index=dataset_index)
data = np.array(data)
data = np.squeeze(data)   # remove singleton dimensions

# Ensure 3D stack (frames, height, width)
if data.ndim == 3:
    num_frames_total = data.shape[0]
    H, W = data.shape[1], data.shape[2]
elif data.ndim == 2:
    num_frames_total = 1
    H, W = data.shape
    data = data[np.newaxis, ...]
else:
    raise RuntimeError(f"Unsupported dataset shape: {data.shape}")

# Apply frame range
if frame_end is None:
    frame_end = num_frames_total
frame_indices = range(frame_start, min(frame_end, num_frames_total))
num_frames = len(frame_indices)
print(f"Dataset has {num_frames_total} frames. Processing frames {frame_start} to {frame_end-1} (total {num_frames})")

# ---- Determine photometric interpretation for TIFF (RGB if draw_contours returns color) ----
# We'll write a test annotation on the first frame to decide
test_frame = data[frame_indices[0]]
im_test = ImageManager(dim_h=W, dim_v=H, roisize=32, Nchannels=1, dtype=np.uint16)
im_test.image[0, ...] = test_frame
im_test.find_object_localmax_NMS(**detection_params)
low = np.percentile(test_frame, 1)
high = np.percentile(test_frame, 99)
test_8bit = ((np.clip(test_frame, low, high) - low) * 255.0 / (high - low)).astype('uint8')
test_annotated = im_test.draw_contours_on_image(test_8bit)
if test_annotated.ndim == 3 and test_annotated.shape[-1] == 3:
    photometric = 'rgb'
else:
    photometric = 'minisblack'
print(f"Detected image type: {photometric}")

# ---- Process all frames and write incrementally to BigTIFF ----
total_start = time.time()

with tifffile.TiffWriter(output_tiff, bigtiff=True) as tif:
    for idx, frame_idx in enumerate(frame_indices):
        print(f"\nProcessing frame {idx+1}/{num_frames} (original index {frame_idx}) ...")
        frame16 = data[frame_idx, ...]

        # 1. Find objects
        im = ImageManager(dim_h=W, dim_v=H, roisize=32, Nchannels=1, dtype=np.uint16)
        im.image[0, ...] = frame16

        t0 = time.time()
        im.find_object_localmax_NMS(**detection_params)
        print(f"   Found {len(im.contours)} objects in {(time.time() - t0)*1000:.2f} ms")

        # 2. Build 8‑bit image with contrast stretch
        low = np.percentile(frame16, 1)
        high = np.percentile(frame16, 99)
        frame_clipped = np.clip(frame16, low, high)
        image8bit = ((frame_clipped - low) * 255.0 / (high - low)).astype('uint8')

        # 3. Draw contours
        t2 = time.time()
        annotated = im.draw_contours_on_image(image8bit)
        print(f"   Drawing contours took {(time.time() - t2)*1000:.2f} ms")

        # 4. Write frame to TIFF (append)
        tif.write(annotated, photometric=photometric)

total_time = time.time() - total_start
print(f"\n✅ Saved {num_frames} annotated frames to {output_tiff}")
print(f"Total processing time: {total_time:.2f} s ({total_time/num_frames:.2f} s/frame)")

# Optional: preview last frame
cv2.imshow("Last annotated frame (preview)", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()