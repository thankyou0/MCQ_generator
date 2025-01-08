import os
import json
import traceback
import pandas as pd
from src.MCQgenerator.utils import read_file,get_table_data
import streamlit as st
from src.MCQgenerator.MCQ_generator import generate_evaluate_chain
from src.MCQgenerator.logger import logging



file_path = r"D:\AI\MCQ_generator\data.txt"  # Adjust the path to your file
with open(file_path, 'r', encoding='utf-8') as file:
    TEXT = file.read()

# Define other parameters for the quiz
NUMBER = 4  # Number of MCQs to generate
SUBJECT = "biology"  # Subject for the quiz
TONE = "simple"  # Tone of the quiz



with open('response.json', 'r') as file:
  RESPONSE_JSON = json.load(file)



st.title("MCQ Generator usining Langchain and Huggingface API")


#Create a form using st.form
with st.form("user_inputs"):
  # File Upload
  uploaded_file = st.file_uploader("Choose a PDF or txt file")

  #input fields
  mcq_count = st.number_input("Number of MCQs", min_value=1, max_value=5, value=4)
  
  #subject
  subject = st.text_input("Insert the subject",max_chars=20)

  #tone
  tone = st.selectbox("Insert the tone", options=["simple","moderate", "complex"])
  
  button = st.form_submit_button("Generate MCQs")
  
  
  if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Generating MCQs"):
      try:
        TEXT = read_file(uploaded_file)
        response = generate_evaluate_chain.invoke({
            "text": TEXT,
            "number": mcq_count,
            "subject": subject,
            "tone": tone,
            "response_json": json.dumps(RESPONSE_JSON)  # Format the response JSON as a string
        })
        quiz = response["quiz"]
        review = response["review"]
        st.write("Generated Quiz:")
        st.write(quiz)
        st.write("Review:")
        st.write(review)
        st.write("Quiz Table:")
        st.write(get_table_data(quiz))
        
        
      except Exception as e:
        logging.error(e)
        st.error("An error occured while generating the MCQs")
        
  elif button and uploaded_file is None:
    st.error("Please upload a file")
  elif button and not mcq_count:
    st.error("Please insert the number of MCQs")
  elif button and not subject:
    st.error("Please insert the subject")
  elif button and not tone:
    st.error("Please insert the tone")
  else:
    pass
  
