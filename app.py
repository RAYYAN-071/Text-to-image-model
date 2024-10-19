import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Function to load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        # Load the pre-trained Stable Diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Ensure GPU is used for faster performance
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to generate image from prompt
def generate_image(pipe, prompt):
    try:
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Streamlit app setup
st.title("Text-to-Image Generator")
st.write("Generate images from text using Stable Diffusion")

# Input for text prompt
prompt = st.text_input("Enter a text prompt", "A futuristic city skyline at sunset")

# Button to generate image
if st.button("Generate Image"):
    st.write("Generating image... Please wait.")

    # Load model
    pipe = load_model()

    # Check if the model is loaded successfully
    if pipe is not None:
        # Use a spinner to show progress
        with st.spinner("Generating image... This might take a while!"):
            image = generate_image(pipe, prompt)
        
        # Display the image if generated
        if image:
            st.image(image, caption=f"Generated from: '{prompt}'", use_column_width=True)

            # Save and provide download link
            img_path = "generated_image.png"
            image.save(img_path)
            with open(img_path, "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
    else:
        st.error("Model loading failed. Please try again later.")

# Display whether GPU is available
if torch.cuda.is_available():
    st.write("Using GPU for generation.")
else:
    st.write("GPU not available. Using CPU, which may be slower.")

    try:
        # Load the pre-trained Stable Diffusion model
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")  # Ensure GPU is used for faster performance
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to generate image from prompt
def generate_image(pipe, prompt):
    try:
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Streamlit app setup
st.title("Text-to-Image Generator")
st.write("Generate images from text using Stable Diffusion")

# Input for text prompt
prompt = st.text_input("Enter a text prompt", "A futuristic city skyline at sunset")

# Button to generate image
if st.button("Generate Image"):
    st.write("Generating image... Please wait.")

    # Load model
    pipe = load_model()

    # Check if the model is loaded successfully
    if pipe is not None:
        # Use a spinner to show progress
        with st.spinner("Generating image... This might take a while!"):
            image = generate_image(pipe, prompt)
        
        # Display the image if generated
        if image:
            st.image(image, caption=f"Generated from: '{prompt}'", use_column_width=True)

            # Save and provide download link
            img_path = "generated_image.png"
            image.save(img_path)
            with open(img_path, "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
    else:
        st.error("Model loading failed. Please try again later.")

# Display whether GPU is available
if torch.cuda.is_available():
    st.write("Using GPU for generation.")
else:
    st.write("GPU not available. Using CPU, which may be slower.")
