# main.py
import logging
import time
from app import main

# Configure logging
logging.basicConfig(filename='main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_sql_query(user_question, context):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./SQL")
        model = AutoModelForCausalLM.from_pretrained("./SQL")

        text = f"""CREATE TABLE stadium (
            stadium_id number,
            location text,
            name text,
            capacity number,
        );

    -- Below are some examples:
    -- {context}
    -- use above examples and provided schema to answer my following question:
    -- question: {user_question}
    answer: SELECT"""

        start_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids

        generated_ids = model.generate(input_ids, max_length=2000)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        Answer = generated_text.split('\n')[-1]
        logging.info("Final Answer: %s", Answer)
        logging.info("Time taken by model: %s", time.time() - start_time)
        return Answer

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

def main():
    try:
        user_question, context = main()
        start_time2 = time.time()
        logging.info(get_sql_query(user_question, context))
        logging.info("Total Time Taken: %s", time.time() - start_time2)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
