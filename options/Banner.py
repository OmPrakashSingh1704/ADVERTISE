from options.Banner_Model.Text2Banner import T2I
from options.Banner_Model.Image2Image import I2I
from options.Banner_Model.Image2Image_2 import I2I_2
import gradio as gr

def TextImage(prompt, width=1024, height=1024, guidance_scale=3.5,
              num_inference_steps=28):
    img = T2I(prompt, width, height, guidance_scale, num_inference_steps)
    return img

# def Image2Image(prompt,image):
#     return I2I(image, prompt)

def Image2Image(
    input_image_editor: dict,
    input_text: str,
    seed_slicer: int,
    randomize_seed_checkbox: bool,
    strength_slider: float,
    num_inference_steps_slider: int,
    progress=gr.Progress(track_tqdm=True)
):return I2I(input_image_editor,input_text,seed_slicer,randomize_seed_checkbox,strength_slider,num_inference_steps_slider)

def Image2Image_2(prompt,image,size,num_inference_steps,guidance_scale):
    return I2I_2(image, prompt,size,num_inference_steps,guidance_scale)