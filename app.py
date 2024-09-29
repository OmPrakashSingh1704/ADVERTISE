import gradio as gr
import numpy as np
from options import Banner, Video
from huggingface_hub import login
import os
login(token=os.getenv("TOKEN"))

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

with gr.Blocks() as demo:
    gr.Markdown("# Create your own Advertisement")
    with gr.Tab("Banner"):
        gr.Markdown("# Take your banner to the next LEVEL!")
        with gr.TabItem("Create your Banner"):
            textInput = gr.Textbox(label="Enter the text to get a good start")
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=8,
                        value=1024,
                    )

                    height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=MAX_IMAGE_SIZE,
                        step=32,
                        value=1024,
                    )

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=15,
                    step=0.1,
                    value=3.5,
                )

                num_inference_steps = gr.Slider(
                    label="Number of Inference Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )

            submit = gr.Button("Submit")
            submit.click(
                fn=Banner.TextImage,
                inputs=[textInput, width, height, guidance_scale, num_inference_steps],
                outputs=gr.Image()
            )

        with gr.TabItem("Edit your Banner"):
            with gr.Row():
                with gr.Column():
                    input_image_editor_component = gr.ImageEditor(
                        label='Image',
                        type='pil',
                        sources=["upload", "webcam"],
                        image_mode='RGB',
                        layers=False,
                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))

                    with gr.Row():
                        input_text_component = gr.Text(
                            label="Prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter your prompt",
                            container=False,
                        )
                        submit_button_component = gr.Button(
                            value='Submit', variant='primary', scale=0)

                    with gr.Accordion("Advanced Settings", open=False):
                        seed_slicer_component = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=42,
                        )

                        randomize_seed_checkbox_component = gr.Checkbox(
                            label="Randomize seed", value=True)

                        with gr.Row():
                            strength_slider_component = gr.Slider(
                                label="Strength",
                                info="Indicates extent to transform the reference `image`. "
                                    "Must be between 0 and 1. `image` is used as a starting "
                                    "point and more noise is added the higher the `strength`.",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.85,
                            )

                            num_inference_steps_slider_component = gr.Slider(
                                label="Number of inference steps",
                                info="The number of denoising steps. More denoising steps "
                                    "usually lead to a higher quality image at the",
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=20,
                            )
                with gr.Column():
                    output_image_component = gr.Image(
                        type='pil', image_mode='RGB', label='Generated image', format="png")
                    with gr.Accordion("Debug", open=False):
                        output_mask_component = gr.Image(
                            type='pil', image_mode='RGB', label='Input mask', format="png")
            with gr.Row():
                submit_button_component.click(
                    fn=Banner.Image2Image,
                    inputs=[
                        input_image_editor_component,
                        input_text_component,
                        seed_slicer_component,
                        randomize_seed_checkbox_component,
                        strength_slider_component,
                        num_inference_steps_slider_component
                    ],
                    outputs=[
                        output_image_component,
                        output_mask_component
                    ]
                )

        with gr.TabItem("Upgrade your Banner"):
            img = gr.Image()
            prompt = gr.Textbox(label="Enter the text to get a good start")
            btn = gr.Button()
            with gr.Accordion("Advanced options", open=False):
                size = gr.Slider(label="Size", minimum=256, maximum=MAX_IMAGE_SIZE, step=8, value=1024)
                num_inference_steps = gr.Slider(label="num_inference_steps", minimum=1, maximum=100,step=1, value=20)
                guidance_scale=gr.Slider(label="guidance_scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            out_img = gr.Image()
            btn.click(Banner.Image2Image_2, [prompt, img,size,num_inference_steps,guidance_scale], out_img)


    with gr.Tab("Video"):
        gr.Markdown("# Create your own Video")
        img=gr.Image()
        btn = gr.Button()
        video=gr.Video()
        btn.click(Video.Video, img, video)

demo.launch()
