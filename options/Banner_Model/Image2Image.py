from typing import Tuple

import random,os
import numpy as np
import gradio as gr
import spaces
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline


MAX_SEED = np.iinfo(np.int32).max
IMAGE_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def resize_image_dimensions(
    original_resolution_wh: Tuple[int, int],
    maximum_dimension: int = IMAGE_SIZE
) -> Tuple[int, int]:
    width, height = original_resolution_wh

    # if width <= maximum_dimension and height <= maximum_dimension:
    #     width = width - (width % 32)
    #     height = height - (height % 32)
    #     return width, height

    if width > height:
        scaling_factor = maximum_dimension / width
    else:
        scaling_factor = maximum_dimension / height

    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    new_width = new_width - (new_width % 32)
    new_height = new_height - (new_height % 32)

    return new_width, new_height


@spaces.GPU(duration=100)
def I2I(
    input_image_editor: dict,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int
):
    if not input_text:
        gr.Info("Please enter a text prompt.")
        return None, None

    image = input_image_editor['background']
    mask = input_image_editor['layers'][0]

    if not image:
        gr.Info("Please upload an image.")
        return None, None

    if not mask:
        gr.Info("Please draw a mask on the image.")
        return None, None
    
    pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(DEVICE)

    width, height = resize_image_dimensions(original_resolution_wh=image.size)
    resized_image = image.resize((width, height), Image.LANCZOS)
    resized_mask = mask.resize((width, height), Image.LANCZOS)

    if randomize_seed_checkbox:
        seed_slicer = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed_slicer)
    result = pipe(
        prompt=input_text,
        image=resized_image,
        mask_image=resized_mask,
        width=width,
        height=height,
        strength=strength_slider,
        generator=generator,
        num_inference_steps=num_inference_steps_slider
    ).images[0]
    print('INFERENCE DONE')
    print(type(result))
    return result, resized_mask