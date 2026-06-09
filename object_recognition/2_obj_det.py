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


frame_index = 0  # ToDo: check frame 0 and 7 

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
im.find_object_localmax(
    channel=0,
    bitdepth=12,
    norm_factor=16,          # 0..4095 -> 0..255
    intensity_threshold=10 # adjust based on brightness   
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

t2= time.time()
annotated_image = im.draw_markers(image8bit, size=1)
print("drawing contours time (ms):", (time.time() - t2)*1000)   

#cv2.imshow("Input Image (8-bit view)", image8bit)
cv2.imshow("Annotated EVs (local maxima)", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
