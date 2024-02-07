import logging
from vector_store import VectorStore
import pandas as pd

logger = logging.getLogger(__name__)

def setup_application():
    uploaded_file = './Training_data.csv'
    data = pd.read_csv(uploaded_file, encoding_errors="ignore")
    cols_to_index = data['instruction']
    
    vector_store = VectorStore()
    vector_store.index_data(data, cols_to_index)
    logger.info("Data indexed successfully.")
    return vector_store

def get_similar_vectors(user_query, vector_store):
    try:
        result_ids_and_scores = vector_store.find_similar_vectors(user_query, 5)
        return result_ids_and_scores
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Set up basic logging configuration
    vector_store = setup_application()
    user_question = "fetch me rule details where exception concept is book"
    context = """
"""
    result_ids_and_scores = get_similar_vectors(user_question, vector_store)
    for uid, score in result_ids_and_scores:
        context += f"question: {data.loc[uid]['instruction']}\nanswer: {data.loc[uid]['output']}\n\n"
        logger.info(f"Vector ID: {uid}, Similarity Score: {score}")
    logger.info("Context: %s", context)
