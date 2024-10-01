# Advertisement Creation Tool with Gradio

This project provides a simple and interactive interface to create personalized **Banners** and **Videos** for advertisements using Gradio. With a range of customization options for banners and video creation, users can generate unique content effortlessly, leveraging advanced AI techniques.

## Features

1. **Banner Creation**:
   - Create banners with customizable width, height, guidance scale, and inference steps.
   - Edit existing banners with image editing tools like brush and layers, alongside text prompts.
   - Upgrade your banner using an image-to-image generation model, with advanced controls like seed randomization and strength adjustment.

2. **Video Creation**:
   - Generate videos based on uploaded images with a single click.
   - Provides an interactive interface for generating ad content in video form.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/OmPrakashSingh1704/ADVERTISE/
   ```

2. Navigate into the project directory:

   ```bash
   cd advertisement-creation-tool
   ```

3. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your Hugging Face token as an environment variable:

   ```bash
   export TOKEN=<your_hugging_face_token>
   ```

5. Run the application:

   ```bash
   python app.py
   ```

## Usage

Once you start the application, it will launch the Gradio interface. You can explore two main tabs: **Banner** and **Video**.

### Banner Tab
- **Create your Banner**: Input text to generate a banner, adjust advanced settings like image size, guidance scale, and number of inference steps, then click submit to get the generated banner.
  
- **Edit your Banner**: Upload an existing image, make edits using the image editor and text prompt, and further customize the generated output through advanced settings.

- **Upgrade your Banner**: Upload an image, enter a prompt, and adjust advanced parameters like size, inference steps, and guidance scale to upgrade the banner.

### Video Tab
- **Create your Video**: Upload an image and click the button to generate a video from the provided image using AI-based generation methods.

## Requirements

- Python 3.8+
- Gradio
- Hugging Face Hub integration
- OpenCV
- PIL (Pillow)
- NumPy

The full list of dependencies can be found in the `requirements.txt`.

## Models Used
This project integrates models from Hugging Face to handle text-to-image generation and image-to-image transformations for banners, as well as image-to-video conversion.

## Advanced Settings

In the banner creation and editing sections, users can fine-tune the output by adjusting:
- **Width** and **Height** of the generated image.
- **Guidance Scale**: The extent to which the prompt influences the output.
- **Number of Inference Steps**: The number of steps for model inference, affecting quality and detail.
- **Seed**: Control the randomness of image generation with a seed value.
- **Randomization**: Option to randomize the seed.

## Future Enhancements

- Additional customization options for video creation.
- Further enhancements in image editing for better control over the ad creation process.
- Integration of more advanced AI models for higher-quality outputs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
