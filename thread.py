import json
import re
import base64
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import whisper
from pytube import YouTube
from transformers import pipeline
import openai
import concurrent.futures

# Define API keys
API_KEY = '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'
ORG_KEY = 'org-Y6SEavpjn8GFtEJs9UYDNz5B'
openai.api_key = API_KEY
openai.organization = ORG_KEY

# Initialize global variables
summarizer = pipeline('summarization')

# Read text from file
def read_text(file_path):
    text = file_path.read().decode("utf-8")
    return text

# Preprocess text
def preprocess_text(text):
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    sentences = nltk.sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return sentences, stop_words

# Extract keywords from sentences
def extract_keywords(sentences, stop_words):
    keywords_dict = defaultdict(list)
    for sentence in sentences:
        keywords = [word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum() and word.lower() not in stop_words]
        for keyword in keywords:
            keywords_dict[keyword].append(sentence)
    return keywords_dict

# Generate summary
def generate_summary(text):
    text = re.sub(r'([.?!])', r'\1<eos>', text)
    sentences = text.split('<eos>')

    max_chunk = 500
    chunks = []
    chunk = []

    for sentence in sentences:
        if len(chunk) + len(sentence.split(' ')) <= max_chunk:
            chunk.extend(sentence.split(' '))
        else:
            chunks.append(chunk)
            chunk = sentence.split(' ')

    chunks.append(chunk)

    summary_futures = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for chunk in chunks:
            summary_futures.append(executor.submit(summarizer, ' '.join(chunk), max_length=200, min_length=50, do_sample=False))

    abstractive_summary = ' '.join([summ.result()[0]['summary_text'] for summ in summary_futures])

    return abstractive_summary

# Generate MCQs
def generate_mcq(keyword, context, model):
    prompt = f"""
    Create a multiple-choice question with 4 options based on the following context:\n{context}\n
    Do not mention any numbering or bullets. Mention the answer in italics at last"""
    response = openai.Completion.create(
        model=model, 
        prompt=prompt, 
        max_tokens=100, 
        temperature=0.7)
    question_data = response.choices[0].text.strip()
    mcq = {
        "question": question_data.split("\n")[0],
        "options": [opt.strip() for opt in question_data.split("\n")[1:]],
        "answer": keyword
    }
    return mcq

def generate_mcqs(keywords_dict, model):
    mcqs = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        mcq_futures = {executor.submit(generate_mcq, keyword, " ".join(keywords_dict[keyword]),model): keyword for keyword in keywords_dict}
        for future in concurrent.futures.as_completed(mcq_futures):
            mcqs.append(future.result())

    return mcqs

# Post-process generated MCQs
def format_mcq(mcq):
    formatted_mcq = {
        'question': mcq['question'],
        'options': mcq['options'][:4],
        'answer': mcq['options'][4].strip('*').strip('_')
    }
    return formatted_mcq

# Download button for text
def download_button(content, filename):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'


def app():
    
    st.set_page_config(page_title="MCQ & Summary Generator App", page_icon=":books:")
    st.title("MCQ & Summary Generator App")
    st.markdown("This app generates multiple-choice questions and summaries based on a given YouTube video URL or uploaded text file.")
    input_type = st.radio("Choose input type", ("YouTube video URL", "Text file"))

    text = "" 
    summary =""

    if input_type == "YouTube video URL":
        youtube_video_url = st.text_input("Enter a YouTube video URL")

        if youtube_video_url:
            try:
                youtube_video_content = YouTube(youtube_video_url, use_oauth=True, allow_oauth_cache=True)
                audio_stream = youtube_video_content.streams[6]
                best_audio = audio_stream.download()
                model = whisper.load_model("base")
                result = model.transcribe(best_audio, verbose=True)

                with open("output.txt", "w") as f:
                    f.write(result['text'])

                with open('output.txt', 'r') as f:
                    text = f.read()

                summary = generate_summary(text)

            except Exception as e:
                st.error(f"Error processing YouTube video: {e}")
                return

    else:  # Text file input
        file_path = st.file_uploader("Upload a text file", type=["txt"])
        if file_path is not None:
            text = read_text(file_path)
            st.success("Text file uploaded successfully!")
            summary = generate_summary(text)
        else:
            return

    # Preprocess text
    sentences, stop_words = preprocess_text(summary)

    # Extract keywords from sentences
    keywords_dict = extract_keywords(sentences, stop_words)

    # Generate MCQs
    model = "text-davinci-003"
    mcqs = generate_mcqs(keywords_dict, model)

    
    st.header("Summary")
    st.write(summary)
    
    # Post-process generated MCQs
    formatted_mcqs = [format_mcq(mcq) for mcq in mcqs]

    # Display MCQs
    st.header("Generated MCQs")
    for mcq_idx, mcq in enumerate(formatted_mcqs, start=1):
        st.markdown(f"{mcq_idx}. **{mcq['question']}**")
        for idx, opt in enumerate(mcq['options'], start=1):
            st.write(f"    {idx}. {opt}")
        st.markdown(f"    *Correct answer: {mcq['answer']}*")
        st.write("")

    # Download buttons
    if summary:
        st.markdown(download_button(summary, 'summary.txt'), unsafe_allow_html=True)

    if formatted_mcqs:
        mcqs_json = json.dumps(formatted_mcqs, indent=2)
        st.markdown(download_button(mcqs_json, 'mcqs.json'), unsafe_allow_html=True)


if __name__ == '__main__':
    app()