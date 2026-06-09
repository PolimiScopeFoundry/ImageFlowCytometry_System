import h5py
import numpy as np
import cv2
from image_data_dvp import ImageManager
from find_h5_dataset import get_h5_datasets
import time

# load dataset from h5
filename = r"C:\Users\YSpol\Desktop\EV_python_code_data\stabilo_beads.h5"
data = get_h5_datasets(filename, dataset_index=0)
data = np.array(data)
data = np.squeeze(data)

# selecting a specific frame from the dataset
frame_index = 59  # 0 for first frame, 49 for the 50th, etc.

if data.ndim == 3:
    num_frames = data.shape[0]
    print(f"Dataset has {num_frames} frames. Using frame index: {frame_index}")
    if frame_index >= num_frames:
        raise ValueError(f"Frame index {frame_index} out of range (0–{num_frames-1})")
    image16bit = data[frame_index, ...]
elif data.ndim == 2:
    image16bit = data
else:
    raise RuntimeError(f"Unsupported dataset shape: {data.shape}")

H, W = image16bit.shape


im = ImageManager(dim_h=W, dim_v=H, roisize=64, Nchannels=1, dtype=np.uint16)
im.image[0, ...] = image16bit 

t0 = time.time()

im.find_object(channel=0, min_object_area=10, max_object_area=2000, norm_factor=4)
#print("Centroids X:", im.cx)
#print("Centroids Y:", im.cy)
#print("Num contours:", len(im.contours))

print("finding object time (ms):", (time.time() - t0)*1000)

t1 = time.time()
contrast=1
image8bit = (image16bit*contrast).astype('uint8') 
print("Conversion time (ms):", (time.time() - t1)*1000)

t2 = time.time()
annotated_image = im.draw_contours_on_image(image8bit)
print("Drawing contours time (ms):", (time.time() - t2)*1000)

cv2.imshow("Input Image (8-bit view)", image8bit)
cv2.imshow("Annotated Image", annotated_image)
cv2.imwrite('output.jpg', image8bit)
cv2.imwrite('output_annotated.jpg', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


