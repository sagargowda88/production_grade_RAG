# vector_store.py
import hashlib
import pickle
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(filename='vector_store.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    def __init__(self, embedding_model='model/', max_workers=4, load_from_cache=True):
        self.vector_data = {}
        self.vector_index = {}
        self.max_workers = max_workers
        self.embedding_model = None  # Initialize as None
        
        if load_from_cache:
            self.load_from_cache()
        
        # Load the embedding model
        self.load_embedding_model(embedding_model)

    def load_embedding_model(self, embedding_model):
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"An error occurred while loading the embedding model: {e}", exc_info=True)

    # Rest of the class methods remain the same...

