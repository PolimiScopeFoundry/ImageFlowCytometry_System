import h5py
import numpy as np
import cv2
from image_data_dvp import ImageManager
from find_h5_dataset import get_h5_datasets
import time


filename = "C:\\Users\\Yoginder Singh\\Downloads\\251204_150955.h5"
data = get_h5_datasets(filename, dataset_index=0)
data = np.array(data)
data = np.squeeze(data)


frame_index = 20 # ToDo: check frame 0 and 7

if data.ndim == 3:
    num_frames = data.shape[0]
    print(f"Dataset has {num_frames} frames. Using frame index: {frame_index}")
    if frame_index >= num_frames:
        raise ValueError(f"Frame index {frame_index} out of range (0–{num_frames-1})")
    frame16 = data[frame_index, ...]
elif data.ndim == 2:
    frame16 = data
else:
    raise RuntimeError(f"Unsupported dataset shape: {data.shape}")

H, W = frame16.shape


im = ImageManager(dim_h=W, dim_v=H, roisize=32, Nchannels=1, dtype=np.uint16)
im.image[0, ...] = frame16

t0 = time.time()
im.find_object_localmax_NMS_v2(
    channel=0,
    bitdepth=12,
    norm_factor=16,
    intensity_threshold=12,  # tune based on SNR
    nms_d_min=7,           #To remove overlaps  # stronger NMS -> fewer duplicates
    R_cluster=100,            # cluster radius (px)
    N_cluster=20,            # neighbors >= 20 => aggregate
    use_image_agg_mask=True,
    bright_thresh=40,
    aggregate_min_area=400,
)

print("Centroids X:", im.cx)
print("Centroids Y:", im.cy)
print("Num EV candidates:", len(im.contours))
print("finding object time (ms):", (time.time() - t0)*1000)


t1 = time.time()
low = np.percentile(frame16, 1)
high = np.percentile(frame16, 99)

frame_clipped = np.clip(frame16, low, high)
image8bit = ((frame_clipped - low) * 255.0 / (high - low)).astype('uint8')
#print("conversion time (ms):", (time.time() - t1)*1000)

t2 = time.time()
annotated_image = im.draw_contours_on_image(image8bit)
print("drawing markers time (ms):", (time.time() - t2)*1000)

# Image saving
output_path = r"C:\Users\YSpol\Desktop\EV_python_code_data\annotated_particles.png"
cv2.imwrite(output_path, annotated_image)

print(f"Saved annotated image at: {output_path}")

cv2.imshow("Annotated EVs (local maxima + NMS v2)", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
