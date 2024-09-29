import spaces
import torch
from controlnet_aux import LineartDetector
from diffusers import ControlNetModel,UniPCMultistepScheduler,StableDiffusionControlNetPipeline
from PIL import Image

device= "cuda" if torch.cuda.is_available() else "cpu"
print("Using device for I2I_2:", device)

@spaces.GPU(duration=100)
def I2I_2(image, prompt,size,num_inference_steps,guidance_scale):
    processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

    checkpoint = "ControlNet-1-1-preview/control_v11p_sd15_lineart"
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "radames/stable-diffusion-v1-5-img2img", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image.resize((size,size))
    image=processor(image)
    generator = torch.Generator(device=device).manual_seed(0)
    image = pipe(prompt+"best quality, extremely detailed", num_inference_steps=num_inference_steps, generator=generator, image=image,negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",guidance_scale=guidance_scale).images[0]
    return image