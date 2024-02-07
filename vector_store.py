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
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_workers = max_workers
        
        if load_from_cache:
            self.load_from_cache()

    def create_1d_string_list(self, data, cols):
        data_rows = data[cols].astype(str).values
        return [" ".join(row) for row in data_rows]

    def get_data_hash(self, data):
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def load_embeddings_from_db(self, data_hash):
        try:
            with open(f"./embedding_storage/{data_hash}.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_embeddings_to_db(self, data_hash, embeddings):
        with open(f"./embedding_storage/{data_hash}.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    def load_from_cache(self):
        data_hash = self.get_data_hash(data)  # Assuming data is accessible
        cached_embeddings = self.load_embeddings_from_db(data_hash)
        if cached_embeddings is not None:
            self.vector_data, self.vector_index = cached_embeddings

    def index_data(self, data, cols):
        try:
            data_hash = self.get_data_hash(data)
            cached_embeddings = self.load_embeddings_from_db(data_hash)
            if cached_embeddings is not None:
                self.vector_data, self.vector_index = cached_embeddings
                return

            data_1d = self.create_1d_string_list(data, cols)
            vectors = [self.embedding_model.encode(text) for text in data_1d]
            self.vector_data = {uid: vector for uid, vector in enumerate(vectors)}
            self._update_index()

            self.save_embeddings_to_db(data_hash, (self.vector_data, self.vector_index))
            logging.info("Data indexed successfully.")

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    def _update_index(self):
        for vector_id, vector in self.vector_data.items():
            self.vector_index[vector_id] = {other_id: self._cosine_similarity(vector, other_vector)
                                            for other_id, other_vector in self.vector_data.items() if vector_id != other_id}

    def find_similar_vectors(self, query_text, num_results=5):
        try:
            query_vector = self.embedding_model.encode(query_text)
            results = []
            for vector_id, vector in self.vector_data.items():
                similarity = self._cosine_similarity(query_vector, vector)
                results.append((vector_id, similarity))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:num_results]

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)

    def _cosine_similarity(self, vector1, vector2):
        try:
            dot_product = np.dot(vector1, vector2)
            norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            similarity = dot_product / (1e-6 + norm_product)
            return similarity
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
