import os
import nltk

# Go one step back from the current directory (src) to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'venv'))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Add this path to nltk's data search paths
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources to the specified directory
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
