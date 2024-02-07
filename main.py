import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from concurrent.futures import ThreadPoolExecutor
from app import user_question, context  # Added import statement

tokenizer = AutoTokenizer.from_pretrained("./SQL")
model = AutoModelForCausalLM.from_pretrained("./SQL")
logger = logging.getLogger(__name__)

def get_sql_query(user_question, context):
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
    logger.info("Final answer: %s", Answer)
    logger.info("Time taken by model: %s", time.time()-start_time)
    return Answer

start_time2 = time.time()
with ThreadPoolExecutor() as executor:
    future = executor.submit(get_sql_query, user_question, context)
    logger.info(future.result())
logger.info("Total Time Taken: %s", time.time()-start_time2)
