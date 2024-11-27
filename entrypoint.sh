#!/bin/sh

# Activate the virtual environment
. /app/venv/bin/activate

# Run the NLTK setup script
python src/setup-nltk.py

# Run the Streamlit app
streamlit run streamlit_app.py --server.port 8502