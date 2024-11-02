import os
import json
import logging
import re
from typing import Dict, Optional
import torch
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.models import ImdbLSTM

# Set up logging configuration with OS-independent path
log_dir = os.path.join('log')
log_file = os.path.join(log_dir, 'src-utils.log')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english')) - {"not", "no", "never", "neither", "none"}
lemmatizer = WordNetLemmatizer()


def load_slang_dictionary() -> Dict[str, str]:
    """
    Load and return the slang dictionary from JSON file.

    Returns:
        Dict[str, str]: A dictionary mapping slang terms to their expanded forms.
    """
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'slang_and_short_forms.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            slang_dict = json.load(file)
        logger.info("Slang dictionary loaded successfully")
        return slang_dict
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding slang dictionary JSON file: {e}")
        return {}


def expand_slang_and_short_forms(text: str, slang_dict: Dict[str, str]) -> str:
    """
    Replace slang and short forms in text with their full forms.

    Args:
        text (str): Input text containing slang terms.
        slang_dict (Dict[str, str]): Dictionary of slang terms and their expansions.

    Returns:
        str: Text with slang terms expanded to their full forms.
    """
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)


def data_preprocessing(review: str) -> str:
    """
    Clean, tokenize, remove stop words, and lemmatize the review text.

    Args:
        review (str): Input review text to be preprocessed.

    Returns:
        str: Preprocessed text with stop words removed and words lemmatized.
    """
    review = re.sub(r'<.*?>', '', review)  # Remove HTML tags
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)  # Remove URLs
    review = re.sub(r'\S+@\S+', '', review)  # Remove emails
    review = re.sub(r'@\w+', '', review)  # Remove mentions
    review = re.sub(r'[^A-Za-z0-9\s]', ' ', review)  # Remove special characters
    review = review.lower()
    tokens = word_tokenize(review)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    processed_review = ' '.join(tokens)
    return processed_review


def preprocess_text(text: str, slang_dict: Dict[str, str]) -> str:
    """
    Perform slang expansion and text preprocessing on the input text.

    Args:
        text (str): Input text to be preprocessed.
        slang_dict (Dict[str, str]): Dictionary of slang terms and their expansions.

    Returns:
        str: Preprocessed text with slang expanded and cleaned.
    """
    logger.info("Preprocessing text")
    text = expand_slang_and_short_forms(text, slang_dict)
    return data_preprocessing(text)


def load_vectorizer() -> Optional[TfidfVectorizer]:
    """
    Load and return the vectorizer from a file.

    Returns:
        Optional[Union[joblib, None]]: Loaded vectorizer if successful; None if loading fails.
    """
    vectorizer_path = os.path.join('checkpoints', 'vectorizer.joblib')
    try:
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at path: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Vectorizer loaded successfully")
        return vectorizer
    except FileNotFoundError as e:
        logger.error(f"Vectorizer file missing: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading vectorizer: {e}")
    return None


def load_model(device: str = 'cpu') -> Optional[ImdbLSTM]:
    """
    Initialize and load the model with weights from the checkpoint.

    Args:
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        Optional[torch.nn.Module]: Loaded model with weights if successful; None if loading fails.
    """
    model_path = os.path.join('checkpoints', 'ImdbLSTM_checkpoint.pth.tar')
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        # Initialize the model architecture
        model = ImdbLSTM(input_size=5000, lstm_hidden_size=130, lstm_layers=3, fc_size=[64, 32, 16], op_size=1).to(device)
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        # Load the state_dict
        model.load_state_dict(checkpoint, strict=False)  # Allow flexibility in loading weights
        model.eval()
        logger.info("Model loaded successfully with weights only")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
    return None


def make_prediction(review_tensor: torch.Tensor, model: torch.nn.Module) -> Optional[str]:
    """
    Generate sentiment prediction from the model.

    Args:
        review_tensor (torch.Tensor): Tensor representation of the review.
        model (torch.nn.Module): Trained model used for prediction.

    Returns:
        Optional[str]: Predicted sentiment ("Positive" or "Negative") if successful; None if an error occurs.
    """
    try:
        review_tensor = review_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(review_tensor)
            prediction = torch.sigmoid(output).item()
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        logger.info(f"Sentiment analysis completed with result: {sentiment}")
        return sentiment
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None


if __name__ == "__main__":
    pass
