import streamlit as st
import PyPDF2
import re
import os
import tempfile
import hashlib
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time
import json
from collections import OrderedDict
import fitz  # PyMuPDF
import pdfplumber
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

# OpenAI client setup
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# PDF Processing Functions
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ')
    return text.strip()

def process_pdf(file_path):
    raw_text = extract_text_from_pdf(file_path)
    processed_text = preprocess_text(raw_text)
    return processed_text

# Highlight text in PDF
def highlight_pdf(pdf_path, highlight_words, output_pdf_path):
    print("highlight_words:", highlight_words)
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        for word in highlight_words:
            normalized_word = word.lower().strip()
            pattern = re.compile(r'\b' + re.escape(normalized_word) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                highlight_rects = page.search_for(match.group(), quads=True)
                for rect in highlight_rects:
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()

    doc.save(output_pdf_path)
    doc.close()
    return output_pdf_path


def match_query_to_text(query, pdf_text):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', pdf_text)
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, sentence_embeddings)
    top_k = min(3, len(sentences))
    top_results = similarities.topk(k=top_k)
    top_indices = top_results[1].flatten().tolist()
    top_sentences = [sentences[idx].strip() for idx in top_indices]
    output = [(query, sentence) for sentence in top_sentences]
    return (output,)

@lru_cache(maxsize=100)
def cached_process_query(query, pdf_text, pdf_file_path):
    return process_query(query, pdf_text, pdf_file_path)

def process_query(query, pdf_text, pdf_file_path):
    relevant_results_tuple = match_query_to_text(query, pdf_text)
    relevant_results = relevant_results_tuple[0]
    
    if not relevant_results:
        return "I couldn't find any relevant information to answer your question.", None

    relevant_text = " ".join([text for name, text in relevant_results])
    all_text = f"{query} {relevant_text}"

    prompt = f"""Based on the following text, provide:

        1. A concise answer to: '{query}'
        2. A Python list of words from your answer

        Text:
        {all_text}

        Answer and Word List:"""

    completion = client.chat.completions.create(
        model="model-identifier", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=False
    )

    response = completion.choices[0].message.content.strip()
    parts = response.split('\n')
    answer = parts[0].strip()
    
    word_list_str = ' '.join(parts[1:])
    word_list_match = re.search(r'\[(.+?)\]', word_list_str)
    
    if word_list_match:
        words_to_highlight = word_list_match.group(1).split(',')
        words_to_highlight = [word.strip().strip("'\"") for word in words_to_highlight]
    else:
        words_to_highlight = []

    common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    words_to_highlight = [word for word in words_to_highlight if word.lower() not in common_words and len(word) > 2]
    
    print(words_to_highlight)

    base_directory = os.path.dirname(pdf_file_path)
    output_pdf_filename = os.path.basename(pdf_file_path).replace(".pdf", "_highlighted.pdf")
    output_pdf_path = os.path.join(base_directory, output_pdf_filename)
    highlight_pdf(pdf_file_path, words_to_highlight, output_pdf_path)

    return answer, output_pdf_path

def process_pdf_content(pdf_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_content)
        tmp_file_path = tmp_file.name
    
    try:
        raw_text = extract_text_from_pdf(tmp_file_path)
        processed_text = preprocess_text(raw_text)
    finally:
        os.remove(tmp_file_path)
    
    return processed_text

# Streamlit App
st.set_page_config(layout="wide", page_title="Chat with PDFs using Local LLM")

st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-color: #4CAF50 !important;  /* Green border */
        box-shadow: none !important;  /* Remove shadow (red outline) */
    }
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

if 'pdf_contents' not in st.session_state:
    st.session_state.pdf_contents = {}
if 'pdf_texts' not in st.session_state:
    st.session_state.pdf_texts = {}
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'clear_query' not in st.session_state:
    st.session_state.clear_query = False
if 'query_input' not in st.session_state:  
    st.session_state.query_input = ""

st.title("Chat with PDFs using Local LLM")

col1, col2 = st.columns([2, 1])

with col1:
    # Chat history
    st.subheader("Chat History")
    chat_history = st.container()
    with chat_history:
        for message in st.session_state.messages:
            st.write(f"{message['role']}: {message['content']}")

    # Query input
    query = st.text_input("Enter your query:", key="query_input", value="" if st.session_state.clear_query else st.session_state.query_input)

    # Query processing
    if st.button("Submit Query"):
        if query:
            st.session_state.clear_query = True  # flag to clear the input after processing
            try:
                with st.spinner("Processing your query..."):
                    responses = []
                    for pdf_name, pdf_text in st.session_state.pdf_texts.items():
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(st.session_state.pdf_contents[pdf_name])
                            tmp_file_path = tmp_file.name

                        try:
                            response, highlighted_pdf_path = cached_process_query(query, pdf_text, tmp_file_path)
                            responses.append((pdf_name, response, highlighted_pdf_path))
                        finally:
                            os.remove(tmp_file_path)

                    final_response = "\n\n".join([f"From {name}: {response}" for name, response, _ in responses])
                    
                
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                    # Display response
                    st.write("Response:", final_response)

                    # Update the session state with highlighted PDF paths
                    st.session_state.highlighted_pdfs = {name: path for name, _, path in responses}

            except Exception as e:
                st.error(f"An error occurred while processing your query: {str(e)}")
        else:
            st.session_state.clear_query = False  # Don't clear input if query is empty

with col2:
    # File uploader
    st.subheader("Choose PDF files")
    uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                pdf_content = uploaded_file.read()
                st.session_state.pdf_contents[uploaded_file.name] = pdf_content
                
                # Process PDF content
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    pdf_text = process_pdf_content(pdf_content)
                    st.session_state.pdf_texts[uploaded_file.name] = pdf_text
                
                st.success(f"{uploaded_file.name} processed and ready for queries.")
                
                # Display PDF preview (either highlighted or original)
                st.subheader(f"Resume: {uploaded_file.name}")
                if uploaded_file.name in st.session_state.get('highlighted_pdfs', {}):
                    pdf_path = st.session_state.highlighted_pdfs[uploaded_file.name]
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600px"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(pdf_content).decode()}" width="100%" height="600px"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")