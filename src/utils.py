import os
import re
import logging
import joblib
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.models import ImdbLSTM
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes raised")


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


# Expand slang using the dynamically loaded dictionary
def expand_slang_and_short_forms(text, slang_dict):
    """Replace slang and short forms in text with their full forms."""
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, slang_dict.keys())) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

def data_preprocessing(review):
    """Preprocess the text data: clean, tokenize, remove stop words, and lemmatize."""
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

def preprocess_text(text, slang_dict):
    """Perform slang expansion and text preprocessing on the input text."""
    logger.info("Preprocessing text")
    text = expand_slang_and_short_forms(text, slang_dict)
    return data_preprocessing(text)

def load_vectorizer():
    """Load and return the vectorizer."""
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

def load_model(device='cpu'):
    """Initialize and load the model with weights from the checkpoint."""
    model_path = os.path.join('checkpoints', 'ImdbLSTM_checkpoint.pth.tar')
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")
        
        # Initialize the model architecture
        model = ImdbLSTM(input_size=5000, lstm_hidden_size=130, lstm_layers=3, fc_size=[64, 32, 16], op_size=1).to(device)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
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


def make_prediction(review_tensor, model):
    """Generate sentiment prediction."""
    try:
        # Ensure the tensor has the correct shape
        review_tensor = review_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(review_tensor)
            prediction = torch.sigmoid(output).item()
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        logger.info(f"Sentiment analysis completed with result: {sentiment}")
        return sentiment
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        
        

# Main script entry point (for testing or standalone use)
if __name__ == "__main__":
    pass