# app.py
import logging
from vector_store import VectorStore
import pandas as pd

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def upload_and_index_data(uploaded_file):
    """
    This function is meant to be used only by developers to upload and index data.
    """
    try:
        data = pd.read_csv(uploaded_file, encoding_errors="ignore")
        cols_to_index = data['instruction']

        vector_store = VectorStore()
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
        if len(sys.argv) > 1 and sys.argv[1] == "--upload-index":
            # Developer mode: upload and index data
            if not upload_and_index_data('./Training_data.csv'):
                return
        else:
            # Regular mode: handle user queries
            if query:
                result_ids_and_scores = vector_store.find_similar_vectors(query, 5)
                result_df = data.loc[[uid for uid, _ in result_ids_and_scores]].reset_index(drop=True)
                for index, row in result_df.iterrows():
                    context += f"question: {row['instruction']}\nanswer: {row['output']}\n\n"
                logging.info("Context generated successfully.")
                logging.info(context)
        return user_question, context

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()



# python app.py --upload-index
# This will execute the upload_and_index_data function and index the data. After the data is indexed, the script will exit, as it's in developer mode. 
# When the script is run without the --upload-index argument, it will operate in regular mode and handle user queries.
