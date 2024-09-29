from huggingface_hub import InferenceClient

def T2I(prompt, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28):
    # Initialize the model client
    model = InferenceClient(model="black-forest-labs/FLUX.1-dev")

    # Prepare the request parameters
    payload = {
        "prompt": prompt,
       "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps
    }

    # Remove None values to avoid sending unsupported arguments
    payload = {k: v for k, v in payload.items() if v is not None}

    # Make the request to generate an image
    return model.text_to_image(**payload)
