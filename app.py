# streamlit_app.py
from vector_store import VectorStore
import pandas as pd

uploaded_file = './Training_data.csv'

data = pd.read_csv(uploaded_file, encoding_errors="ignore")

# Allow the user to select a specific column for indexing
cols_to_index = data['instruction']

vector_store = VectorStore()
vector_store.index_data(data, cols_to_index)
print("Data indexed successfully!")

user_question = "fetch me rule details where exception concept is book"
query = user_question
context = """
"""
if query:
    try:
        # result_ids = [uid for uid, _ in vector_store.find_similar_vectors(query, 5)]
        # result_df = data.loc[result_ids].reset_index(drop=True)
        # st.write(result_df)
        result_ids_and_scores = vector_store.find_similar_vectors(query, 5)
        result_df = data.loc[[uid for uid, _ in result_ids_and_scores]].reset_index(drop=True)
        # print(result_df)
        for index,row in result_df.iterrows():
            context += f"question: {row['instruction']}\nanswer: {row['output']}\n\n"
        # result_df["Similarity Score"] = [score for _, score in result_ids_and_scores]
        
    except Exception as e:
        print(f"An error occurred: {e}")

print(context)
