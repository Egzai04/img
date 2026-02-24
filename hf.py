import streamlit as st
from huggingface_hub import InferenceClient

hf_token = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="Hugging Face Image Generator", page_icon="🎨")
st.title("Egzai's Image Generator")

prompt = st.text_input("Describe the image you want:")

if st.button("Generate"):
    if not prompt:
        st.warning("Please enter a prompt")
    else:
        try:
            with st.spinner("Generating image..."):

                client = InferenceClient(
                    provider="hf-inference",
                    api_key=hf_token
                )

                image = client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-schnell"
                )

                st.image(image, caption="Generated Image", use_container_width=True)

        except Exception as e:
            st.error(str(e))
