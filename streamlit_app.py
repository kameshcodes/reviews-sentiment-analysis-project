import os
import logging
from typing import Optional
import torch
import streamlit as st
from src.utils import (preprocess_text,
                       load_vectorizer,
                       load_model,
                       make_prediction,
                       load_slang_dictionary)

# Logging configuration
log_dir = os.path.join('log')
log_file = os.path.join(log_dir, 'app.log')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Sentiment Analysis Project",
    page_icon="😇",
    layout="centered",
)

# Hide Streamlit elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    div[data-testid="stStatusWidget"] {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_image_base64(image_path: str) -> str:
    """
    Encode an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def display_flow_image() -> None:
    """
    Display the application development flow image in the Streamlit app.
    """
    image_path = os.path.join("imgs", "dev-flow.png")

    if os.path.exists(image_path):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{get_image_base64(image_path)}"
                alt="Application Development Flow"
                style="width: 100%; height: 70%;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Image 'dev-flow.png' not found.")
        logger.error("File 'dev-flow.png' not found. Cannot display application development flow image.")


def perform_sentiment_analysis(text: str, device: str = 'cpu') -> Optional[str]:
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The input text to analyze.
        device (str): The device to use for the model ('cpu' or 'cuda').

    Returns:
        Optional[str]: The sentiment prediction ("Positive" or "Negative"), or an error message if any issue arises.
    """
    logger.info("Starting sentiment analysis")
    slang_dict = load_slang_dictionary()
    preprocessed_text = preprocess_text(text, slang_dict)
    vectorizer = load_vectorizer()

    if not vectorizer:
        logger.error("Failed to load vectorizer")
        return "Error loading vectorizer"

    try:
        review_vec = vectorizer.transform([preprocessed_text])
        review_tensor = torch.tensor(review_vec.toarray(), dtype=torch.float32).to(device)
        logger.info("Text converted to tensor successfully")
    except Exception as e:
        logger.error(f"Error transforming text to tensor: {e}")
        return f"Error transforming text to tensor: {e}"

    model = load_model(device)
    if not model:
        logger.error("Failed to load model")
        return "Error loading model"

    return make_prediction(review_tensor, model)


# Main UI setup
st.markdown(
    """
    <h1 style='position: fixed; top: 6.5%; left: 8.5%; transform: translateX(-50%); color: #316FF6; padding: 5px;'>
    Projects
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h5 style='position: fixed; top: 13%; left: 14%; transform: translateX(-50%); color: #316FF6; padding: 5px;'>
    by Kamesh Dubey
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="display: flex; gap: 10px; margin-bottom: 30px; position: fixed; top: 9%; left: 16%; transform: translateX(50%);">
        <a href="?page=home" style="background-color: #535353; color: white; padding: 3px 20px; border-radius: 5px; text-decoration: none; font-size: 16px;">Home</a>
        <a href="?page=app" style="background-color: #535353; color: white; padding: 3px 20px; border-radius: 5px; text-decoration: none; font-size: 16px;">App</a>
        <a href="https://github.com/kameshcodes/reviews-sentiment-analysisi-project" target="_blank" style="background-color: #535353; color: white; padding: 5px 20px; border-radius: 5px; text-decoration: none; font-size: 16px;">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True
)

if 'page' not in st.session_state:
    st.session_state['page'] = 'app'

page = st.query_params.get("page", ["app"])[0]
st.session_state['page'] = page

if st.session_state['page'] == 'home':
    st.title("Sentiment Analysis App")
    st.write(
        "- **Text Vectorizer**: TF-IDF\n"
        "- **Underlying Model**: LSTM"
    )
    display_flow_image()
    st.empty()

elif st.session_state['page'] == 'app':
    st.title("Sentiment Analysis App")
    st.markdown("""
    - **Enter Review**: Type or paste your review (minimum 10 words required).
    - **Click "Analyze"**: To start the sentiment analysis.
    - **Word Check**: A warning appears if fewer than 10 words or if text is empty.
    """)

    text_input = st.text_area("Enter your review for sentiment analysis (minimum 10 words required):", "")
    word_count = len(text_input.split())

    if st.button("Analyze"):
        if word_count >= 10:
            with st.spinner("Analyzing sentiment..."):
                logger.info(f"User input received for analysis: {text_input}")
                sentiment = perform_sentiment_analysis(text_input)
                st.success(f"Sentiment: {sentiment}")
        elif word_count > 0:
            st.warning("Please enter at least 10 words for analysis.")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    pass
