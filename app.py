import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the pre-trained Stable Diffusion model
@st.cache(allow_output_mutation=True)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Ensure GPU is used for better performance
    return pipe

# Function to generate image from prompt
def generate_image(pipe, prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    return image

# Streamlit app setup
st.title("Text-to-Image Generator")
st.write("Generate images from text using Stable Diffusion")

# Input for text prompt
prompt = st.text_input("Enter a text prompt", "A futuristic city skyline at sunset")

# Button to generate image
if st.button("Generate Image"):
    st.write("Generating image...")
    
    # Load model and generate image
    pipe = load_model()
    image = generate_image(pipe, prompt)
    
    # Display the generated image
    st.image(image, caption=f"Generated from: '{prompt}'", use_column_width=True)

    # Option to download the image
    img_path = "generated_image.png"
    image.save(img_path)
    with open(img_path, "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name="generated_image.png",
            mime="image/png"
        )
