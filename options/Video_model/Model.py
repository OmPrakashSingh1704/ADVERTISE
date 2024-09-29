import spaces
import torch,os,imageio
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from PIL import Image
from glob import glob
from pathlib import Path
import numpy as np

# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_video(frames, save_path, fps, quality=9):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

# Function to generate the video
@spaces.GPU(duration=100)
def Video(image):


    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16
    ).to(device)

    # Enable model offloading if using the CPU
    if device == "cpu":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.enable_sequential_cpu_offload()

    
    image = Image.fromarray(image)
    image = image.resize((1024, 576))

    # Set random seed for reproducibility
    generator = torch.manual_seed(42)
    output_folder= "outputs"
    os.makedirs(output_folder, exist_ok=True)
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
    # Generate the video frames
    frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
    # Export the frames to a video file
    export_to_video(frames, video_path, fps=7)
    return video_path