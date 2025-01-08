import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
# from src.MCQgenerator.utils import read_file, get_table_data
# from src.MCQgenerator.logger import logging


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import SequentialChain


load_dotenv()

key = os.getenv("HUGGINGFACE_API_KEY")



llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=key,
    temperature=0.7  # Explicit temperature setting
)



TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming to the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs.
here is my response structure, i want response to be in json with given structure.
Strictly don't give any other text other than given below response_json format in response which is json format.
{response_json}
"""


quiz_generation_prompt = PromptTemplate(input_variables=["text", "number", "subject", "tone", "response_json"],template=TEMPLATE)


quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=False)





template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template2)


review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=False)




# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=False,)




# file_path = r"D:\AI\MCQ_generator\data.txt"  # Adjust the path to your file
# with open(file_path, 'r', encoding='utf-8') as file:
#     TEXT = file.read()

# # Define other parameters for the quiz
# NUMBER = 3  # Number of MCQs to generate
# SUBJECT = "biology"  # Subject for the quiz
# TONE = "simple"  # Tone of the quiz
# RESPONSE_JSON = {
#     "1": {
#         "mcq": "multiple choice question",
#         "options": {
#             "a": "choice here",
#             "b": "choice here",
#             "c": "choice here",
#             "d": "choice here",
#         },
#         "correct": "correct answer",
#     },
#     "2": {
#         "mcq": "multiple choice question",
#         "options": {
#             "a": "choice here",
#             "b": "choice here",
#             "c": "choice here",
#             "d": "choice here",
#         },
#         "correct": "correct answer",
#     }
# }



# # Run the quiz generation and evaluation chain
# response = generate_evaluate_chain.invoke({
#     "text": TEXT,
#     "number": NUMBER,
#     "subject": SUBJECT,
#     "tone": TONE,
#     "response_json": json.dumps(RESPONSE_JSON)  # Format the response JSON as a string
# })


# # Extract the outputs
# quiz = response["quiz"]
# review = response["review"]

# print("Generated Quiz:")
# print(quiz)


# print("\n\nReview:")
# print(review)