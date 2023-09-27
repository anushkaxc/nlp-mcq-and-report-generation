import streamlit as st
from PyPDF2 import PdfReader #library used to deal read pdfs
import re  #library used to deal with regular expressions
#regular expressions are used to defines certain number of sets in algebraic fashion
from textblob import TextBlob
import random


#this function is made to get extract text from any pdf
def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    print("The number of pages in this pdf file are", len(reader.pages))
    text=""
    page = reader.pages[0]
    text = page.extract_text()
   #st.write(text)
    print("----------------------------------------------------------")
    text = text.replace("\n", " ")
    #st.write(text)
    return text

def line(pdf_path):
    reader=PdfReader(pdf_path)
    l=len(reader.pages)
    return l    

st.title("PDF Text Summarizer")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def generate_mcqs(text, num_options=4):
    blob = TextBlob(text)
    sentences = blob.sentences

    mcqs = []

    for sentence in sentences:
        question = sentence.replace('.', '?')  # Turn statement into a question

        options = [sentence]

        while len(options) < num_options:
            random_sentence = random.choice(sentences)
            if random_sentence != sentence:
                options.append(random_sentence)

        random.shuffle(options)

        mcq = {
            'question': question,
            'options': options,
            'answer': options.index(sentence)
        }

        mcqs.append(mcq)

    return mcqs


if uploaded_file is not None:
    st.write("### Uploaded PDF:")
    st.write(uploaded_file.name)

    text = pdf_to_text(uploaded_file)  #the function is called to extract to text from uploaded pdf
    st.write("### Extracted Text:")
    st.write(text)
    print(text)

    lines=line(uploaded_file)
    st.write("The number of pages in this pdf are", lines)

    st.write("MCQs")
    mcqs = generate_mcqs(text)
    for idx, mcq in enumerate(mcqs, 1):
        st.write(f"Question {idx}: {mcq['question']}")
        for i, option in enumerate(mcq['options']):
            st.write(f"{chr(ord('A')+i)}. {option}")
        st.write(f"Answer: {chr(ord('A')+mcq['answer'])}\n")

