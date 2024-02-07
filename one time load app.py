# app.py
import sys
import logging
from vector_store import VectorStore
import pandas as pd

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Instantiate VectorStore with loading from cache
vector_store = VectorStore(load_from_cache=True)

def _upload_and_index_data(uploaded_file):
    """
    This function is meant to be used only by the developer to upload and index data.
    """
    try:
        data = pd.read_csv(uploaded_file, encoding_errors="ignore")
        cols_to_index = data['instruction']

        # Use the global vector_store
        global vector_store
        vector_store.index_data(data, cols_to_index)
        logging.info("Data uploaded and indexed successfully.")
        return True

    except Exception as e:
        logging.error(f"An error occurred during data upload and indexing: {e}", exc_info=True)
        return False

def main():
    user_question = "fetch me rule details where exception concept is book"
    query = user_question
    context = ""
    try:
        # Regular mode: handle user queries
        if query:
            data = pd.read_csv('./Training_data.csv', encoding_errors="ignore")
            result_ids_and_scores = vector_store.find_similar_vectors(query, 5)  # Reuse vector_store
            result_df = data.loc[[uid for uid, _ in result_ids_and_scores]].reset_index(drop=True)
            for index, row in result_df.iterrows():
                context += f"question: {row['instruction']}\nanswer: {row['output']}\n\n"
            logging.info("Context generated successfully.")
            logging.info(context)
        return user_question, context

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

# Call the _upload_and_index_data function only in developer mode
if len(sys.argv) > 1 and sys.argv[1] == "--upload-index":
    _upload_and_index_data('./Training_data.csv')

if __name__ == "__main__":
    main()
