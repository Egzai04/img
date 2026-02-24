import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Page config
st.set_page_config(page_title="Hugging Face Image Generator", page_icon="🎨")
st.title("Egzai's Image Generator")

prompt = st.text_input("Describe the image you want:")

if st.button("Generate"):
    if not hf_token:
        st.error("HF_TOKEN not found in .env file")
    elif not prompt:
        st.warning("Please enter a prompt")
    else:
        try:
            with st.spinner("Generating image..."):

                # Create client using HF free inference provider
                client = InferenceClient(
                    provider="hf-inference",
                    api_key=hf_token
                )

                # Generate image
                image = client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-schnell"
                )

                # Display image
                st.image(image, caption="Generated Image", use_container_width=True)

        except Exception as e:
            st.error("Error occurred:")
            st.write(str(e))