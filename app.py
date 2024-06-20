import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template
import pickle
import io
import logging
import time

# Download stopwords
nltk.download('stopwords')

# Create Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load trained model and TF-IDF vectorizer
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")

try:
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load vectorizer: {e}")

# Define the stemming function
stemming = PorterStemmer()

def port_stemming(contents):
    stem_words = re.sub('[^a-zA-Z]', ' ', contents)
    stem_words = stem_words.lower()
    stem_words = stem_words.split()
    stem_words = [stemming.stem(word) for word in stem_words if word not in stopwords.words('english')]
    stem_words = ' '.join(stem_words)
    return stem_words

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        logging.warning("No file part in the request.")
        return render_template("index.html", prediction_text="No file part")
    
    file = request.files['file']
    chunk_size = request.form.get('chunk_size')
    
    if chunk_size == '' or chunk_size is None:
        chunk_size = 1000  # Default chunk size for testing
    else:
        try:
            chunk_size = int(chunk_size)
        except ValueError:
            logging.warning("Invalid chunk size provided.")
            return render_template("index.html", prediction_text="Invalid chunk size provided")
    
    if file.filename == '':
        logging.warning("No selected file.")
        return render_template("index.html", prediction_text="No selected file")
    
    if file:
        try:
            # Save the uploaded file in memory
            file_bytes = file.read()
            file_buffer = io.BytesIO(file_bytes)
            
            # Read the entire Parquet file into a DataFrame
            df = pd.read_parquet(file_buffer, engine='auto')
            
            if 'text' not in df.columns:
                logging.warning("Parquet file does not contain 'text' column.")
                return render_template("index.html", prediction_text="Parquet file does not contain 'text' column")
            
            # Drop rows with missing text data
            df.dropna(subset=['text'], inplace=True)
            
            # Shuffle the DataFrame
            df = df.sample(frac=1).reset_index(drop=True)
            
            # Select a random chunk of the data
            chunk = df.head(chunk_size)
            text_data = chunk['text'].tolist()
                
            # Log the text data for debugging
            logging.debug(f"Processing chunk of size: {len(text_data)}")
                
            # Apply stemming
            start_time = time.time()
            text_data_stemmed = [port_stemming(text) for text in text_data]
            logging.debug(f"Stemming time: {time.time() - start_time}s")
                
            # Transform text data to TF-IDF vectors
            start_time = time.time()
            try:
                text_data_transformed = vectorizer.transform(text_data_stemmed)
                logging.debug(f"Vectorization time: {time.time() - start_time}s")
            except Exception as e:
                logging.error(f"Vectorization error: {e}")
                return render_template("index.html", prediction_text=f"Error in vectorization: {str(e)}")
                
            # Ensure model receives TF-IDF vectors for prediction
            start_time = time.time()
            try:
                predictions = model.predict(text_data_transformed)
                logging.debug(f"Prediction time: {time.time() - start_time}s")
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                return render_template("index.html", prediction_text=f"Error in prediction: {str(e)}")
                
            # Map predictions to sentiment labels
            sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
            predictions = [sentiment_mapping.get(pred, 'Unknown') for pred in predictions]
                
            chunk_results = pd.DataFrame({'text': chunk['text'], 'sentiment': predictions})
            result_html = chunk_results.to_html(index=False)
            logging.debug(f"Result HTML: {result_html}")
            
            return render_template("index.html", prediction_text=result_html)
        
        except Exception as e:
            logging.error(f"Error during processing: {e}")
            return render_template("index.html", prediction_text=f"Error processing file: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
