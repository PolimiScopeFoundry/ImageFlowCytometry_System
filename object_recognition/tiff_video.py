import numpy as np
import cv2
import tifffile
from image_data_dvp import ImageManager
import time
import os
from datetime import datetime

# ================== CONFIGURATION ==================
input_tiff = "substack_(52-64).tif"      # <-- change to your TIFF file
output_tiff = "substack_annotated.tif"          # output annotated stack
output_video = "substack_video.mp4"         # output video file

# Frame range (optional)
frame_start = 0
frame_end = None          # None = all frames

# Detection parameters (tuned for your beads)
detection_params = {
    "channel": 0,
    "bitdepth": 12,
    "norm_factor": 16,
    "intensity_threshold": 10,
    "nms_d_min": 7,
    "R_cluster": 100,
    "N_cluster": 20
}

# Video settings
video_duration_seconds = 10   # desired video length (auto-calculates FPS)
video_quality = 'high'        # 'high' or 'lossless'
# ===================================================

# ---- 1. Load TIFF stack ----
print(f"Loading TIFF stack from: {input_tiff}")
data = tifffile.imread(input_tiff)

# Ensure 3D stack (frames, height, width)
if data.ndim == 2:
    # Single image – add frame dimension
    data = data[np.newaxis, ...]
elif data.ndim == 3:
    # Already a stack – good
    pass
else:
    raise RuntimeError(f"Unsupported TIFF dimensions: {data.ndim}")

num_frames_total = data.shape[0]
H, W = data.shape[1], data.shape[2]
print(f"Loaded {num_frames_total} frames, size {W}x{H}")

# ---- 2. Apply frame range ----
if frame_end is None:
    frame_end = num_frames_total
frame_indices = range(frame_start, min(frame_end, num_frames_total))
num_frames = len(frame_indices)
print(f"Processing frames {frame_start} to {frame_end-1} (total {num_frames})")

# ---- 3. Determine TIFF photometric interpretation (RGB or grayscale) ----
# Process a test frame to see what draw_contours_on_image returns
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
    print("Detected color output (RGB)")
else:
    photometric = 'minisblack'
    print("Detected grayscale output")

# ---- 4. Process frames and write annotated TIFF incrementally ----
total_start = time.time()

with tifffile.TiffWriter(output_tiff, bigtiff=True) as tif:
    for idx, frame_idx in enumerate(frame_indices):
        print(f"\nFrame {idx+1}/{num_frames} (original index {frame_idx}) ...")
        frame16 = data[frame_idx, ...].astype(np.uint16)

        # Create ImageManager for this frame
        im = ImageManager(dim_h=W, dim_v=H, roisize=32, Nchannels=1, dtype=np.uint16)
        im.image[0, ...] = frame16

        # Detect objects
        t0 = time.time()
        im.find_object_localmax_NMS(**detection_params)
        print(f"   Found {len(im.contours)} objects in {(time.time() - t0)*1000:.1f} ms")

        # Convert to 8-bit with contrast stretch
        low = np.percentile(frame16, 1)
        high = np.percentile(frame16, 99)
        frame_clipped = np.clip(frame16, low, high)
        image8bit = ((frame_clipped - low) * 255.0 / (high - low)).astype('uint8')

        # Draw contours
        t2 = time.time()
        annotated = im.draw_contours_on_image(image8bit)
        print(f"   Drawing took {(time.time() - t2)*1000:.1f} ms")

        # Write to TIFF
        tif.write(annotated, photometric=photometric)

total_time = time.time() - total_start
print(f"\n✅ Annotated TIFF saved to: {os.path.abspath(output_tiff)}")
print(f"Total processing time: {total_time:.1f} s ({total_time/num_frames:.2f} s/frame)")

# ---- 5. Convert annotated TIFF to MP4 video with desired duration ----
print("\n" + "="*50)
print("Converting annotated stack to video...")

# Read the just-saved annotated stack
annotated_stack = tifffile.imread(output_tiff)

# Handle color/gray
if annotated_stack.ndim == 3:
    # Grayscale: convert to 3-channel for video
    video_frames = np.stack([annotated_stack] * 3, axis=-1)
elif annotated_stack.ndim == 4 and annotated_stack.shape[3] == 3:
    # Already RGB (OpenCV needs BGR)
    video_frames = annotated_stack[..., ::-1]  # RGB -> BGR
else:
    raise ValueError(f"Unexpected annotated stack shape: {annotated_stack.shape}")

num_frames_vid = video_frames.shape[0]
height, width = video_frames.shape[1], video_frames.shape[2]

# Calculate FPS to achieve desired duration
fps = num_frames_vid / video_duration_seconds
print(f"Video: {num_frames_vid} frames, target duration {video_duration_seconds}s → FPS = {fps:.2f}")

# Setup codec
if video_quality == 'lossless':
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
else:
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not out.isOpened():
    # Fallback to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write frames
for i in range(num_frames_vid):
    frame = video_frames[i].astype(np.uint8)
    out.write(frame)

out.release()
print(f"✅ Video saved to: {os.path.abspath(output_video)} (duration = {num_frames_vid/fps:.1f}s)")

# ---- 6. Optional: preview last frame ----
cv2.imshow("Last annotated frame (press any key)", annotated_stack[-1])
cv2.waitKey(0)
cv2.destroyAllWindows()