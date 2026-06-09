import tifffile
import cv2
import numpy as np
import os

def tiff_stack_to_mp4(input_tiff_path, output_mp4_path, fps=30, quality='high'):
    """
    Convert a TIFF stack to an MP4 video file.
    
    Args:
        input_tiff_path (str): Path to the input TIFF stack file.
        output_mp4_path (str): Path for the output MP4 file.
        fps (int): Frames per second for the output video.
        quality (str): 'lossless' for best quality (larger file) or 'high' for a good balance.
    """
    # Read the TIFF stack
    print(f"Reading TIFF stack from {input_tiff_path}...")
    img_stack = tifffile.imread(input_tiff_path)
    
    # Check if the stack has 3 dimensions (frames, height, width, channels)
    if img_stack.ndim not in [3, 4]:
        raise ValueError(f"Unsupported image dimensions: {img_stack.ndim}")
    
    # Handle different channel arrangements
    if img_stack.ndim == 4:
        # Assuming (frames, height, width, channels) -> convert BGR to RGB if needed
        if img_stack.shape[3] == 3:
            # OpenCV uses BGR, convert if your TIFF is RGB
            img_stack = img_stack[..., ::-1]
    else:
        # Convert grayscale to 3-channel for video
        img_stack = np.stack([img_stack] * 3, axis=-1)
    
    # Get dimensions
    num_frames, height, width, channels = img_stack.shape
    print(f"Stack info: {num_frames} frames, {width}x{height}, {channels} channels")
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    
    # Adjust quality settings based on user preference
    if quality == 'lossless':
        # Lossless encoding (larger file)
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # FFV1 lossless codec
        # Note: Lossless codecs might not be supported in all video players.
    elif quality == 'high':
        # High quality, good compression
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open VideoWriter. Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise IOError(f"Failed to create video at {output_mp4_path}")
    
    # Write each frame to the video
    print(f"Converting to video ({fps} fps)...")
    for i in range(num_frames):
        frame = img_stack[i].astype(np.uint8)
        out.write(frame)
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{num_frames} frames...")
    
    # Release the VideoWriter
    out.release()
    print(f"Success! Video saved to: {output_mp4_path}")

# --- Usage Example ---
if __name__ == "__main__":
    # Replace with the path to your annotated TIFF stack
    input_tiff = "annotated_stack_original.tif"
    # Choose your output MP4 file path
    output_mp4 = "annotated_video_original.mp4"
    
    tiff_stack_to_mp4(input_tiff, output_mp4, fps=10, quality='high')