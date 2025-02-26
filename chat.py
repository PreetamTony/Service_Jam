import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import pytesseract
import re
import json
import hashlib
import sqlite3
import requests
from PIL import Image
from streamlit_mic_recorder import mic_recorder  # For voice input in the general chatbot

# Configuration
GROQ_API_KEY = "groq_api_key"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

##############################################
# Database Functions (Storing School Materials)
##############################################
def init_db():
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS school_materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            summary TEXT,
            extracted_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def generate_file_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def store_school_material(file_hash, summary, extracted_text):
    conn = sqlite3.connect("documents.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO school_materials (file_hash, summary, extracted_text) VALUES (?, ?, ?)",
                  (file_hash, summary, extracted_text))
        conn.commit()
        st.success("✅ School material saved to database!")
    except sqlite3.IntegrityError:
        st.warning("⚠ Duplicate document detected! Not saving to database.")
    conn.close()

##############################################
# PDF Text Extraction Functions
##############################################
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert("L")  # Convert to grayscale for better OCR
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"
    return full_text.strip()

##############################################
# AI Summarization & Quiz Generation Functions
##############################################
def summarize_school_material(text):
    prompt = (
        f"Summarize the following school material in simple language suitable for students. "
        f"Include key concepts, step-by-step explanations, examples, and if possible, links to additional study materials:\n\n{text}"
    )
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an AI tutor summarizing academic materials."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No summary generated.")
    else:
        return f"Error: {response.status_code}, {response.text}"

def generate_quiz(school_summary):
    quiz_prompt = (
        f"Based on the following school material summary, generate an engaging quiz for students with 3 multiple-choice questions. "
        f"Each question should have 4 options, indicate the correct answer, and include a detailed explanation for that answer:\n\n{school_summary}\n\n"
        f"Return the quiz as a JSON object in the format:\n"
        f'{{"questions": [{{"question": "string", "options": ["string", "string", "string", "string"], "correct_answer": "string", "explanation": "string"}}]}}'
    )
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an AI tutor creating quizzes for academic materials."},
            {"role": "user", "content": quiz_prompt}
        ],
        "temperature": 0.9,
        "max_tokens": 1024
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        try:
            return json.loads(response.json()["choices"][0]["message"]["content"])
        except json.JSONDecodeError:
            return None
    else:
        return None

##############################################
# General Academic Chatbot Function
##############################################
def get_detailed_explanation(user_input, student_age, language="English"):
    prompt = f"""You are an AI tutor for students aged {student_age}.
- Explain the concept in **simple language**.
- Provide **step-by-step breakdowns**.
- Include **real-life applications, fun facts, and interesting stories**.
- If there are any **relevant equations**, display them.
- If diagrams would help, **provide a textual description** and a **link to an illustrative image**.
- Include **related study materials** (summaries, PDF links, video references).
- Respond in {language}.

Question: {user_input}
"""
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 2048
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: Unable to fetch response. Please try again."

##############################################
# Streamlit UI Setup & Custom CSS
##############################################
st.set_page_config(page_title="AI-Powered Student Learning", page_icon="📚", layout="wide")

st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTextInput input {
        padding: 10px;
        font-size: 16px;
    }
    .stRadio div {
        flex-direction: row;
    }
    .stMarkdown h3 {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar: Student Details & About
st.sidebar.header("🎓 Student Details")
student_name = st.sidebar.text_input("Enter your name")
student_age = st.sidebar.slider("Select your age", 8, 18, 12)
language = st.sidebar.selectbox("Preferred Language", ["English", "Spanish", "French", "Hindi"])

st.sidebar.markdown("---")
st.sidebar.markdown("## About This App")
st.sidebar.markdown(
    """
    This tool extracts and summarizes school material from PDFs, providing simple explanations and interactive quizzes.
    - Supports digital and scanned PDFs (OCR included)
    - Generates summaries and quizzes powered by AI
    - Includes a dedicated general academic chatbot for all questions
    """
)

# Initialize the database (for storing school materials)
init_db()

# Main Title
st.title("📚 AI-Powered Student Learning")
st.markdown("Upload a school material PDF to extract key learning points and get an AI-powered summary with quiz features!")

##############################################
# File Upload & Text Extraction (School Material)
##############################################
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
school_text = ""
summary = ""

if uploaded_file:
    file_path = "uploaded_file.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("🔍 Extracting text from PDF...")
    school_text = extract_text_from_pdf(file_path)
    # If no text was extracted, assume a scanned document and use OCR
    if not school_text.strip():
        school_text = extract_text_from_scanned_pdf(file_path)

    if school_text:
        st.success("✅ Text extracted successfully!")
        with st.expander("Show Extracted Text"):
            st.text_area("Extracted Text:", school_text, height=250)
    else:
        st.error("Unable to extract text from the PDF.")

##############################################
# Summarization & Storing the Document
##############################################
if school_text:
    with st.spinner("Generating summary..."):
        summary = summarize_school_material(school_text)
    st.subheader("📖 Document Summary")
    st.write(summary)
    # Store the school material in the database
    store_school_material(generate_file_hash(school_text), summary, school_text)

##############################################
# Quiz Generation Section (Based on Document Summary)
##############################################
if summary:
    st.markdown("---")
    st.subheader("📝 Test Your Knowledge!")
    if st.button("Generate Quiz"):
        quiz = generate_quiz(summary)
        if quiz:
            st.session_state.quiz_data = quiz
            st.session_state.user_answers = {i: None for i in range(len(quiz["questions"]))}
            st.experimental_rerun()
        else:
            st.warning("Could not generate a quiz. Please try again!")

    if "quiz_data" in st.session_state and st.session_state.quiz_data:
        st.markdown("### Quiz Time!")
        for i, q in enumerate(st.session_state.quiz_data["questions"]):
            st.write(f"**Q{i+1}: {q['question']}**")
            selected_option = st.radio(
                f"Select your answer for Q{i+1}:", q["options"], index=0, key=f"q{i}"
            )
            st.session_state.user_answers[i] = selected_option

        if st.button("Submit Quiz"):
            correct_count = 0
            total_questions = len(st.session_state.quiz_data["questions"])
            for i, q in enumerate(st.session_state.quiz_data["questions"]):
                user_answer = st.session_state.user_answers.get(i)
                st.write(f"**Q{i+1}: {q['question']}**")
                st.write(f"📝 Your Answer: {user_answer}")
                if user_answer == q["correct_answer"]:
                    st.success("✅ Correct!")
                    correct_count += 1
                else:
                    st.error(f"❌ Incorrect! Correct Answer: {q['correct_answer']}")
                st.info(f"📘 Explanation: {q['explanation']}")
            st.subheader(f"📊 Your Score: {correct_count}/{total_questions}")
            if st.button("🔄 Try Another Quiz"):
                st.session_state.quiz_data = None
                st.session_state.user_answers = {}
                st.experimental_rerun()

##############################################
# Document-Based Q&A Chatbot (Based on PDF Summary)
##############################################
if summary:
    st.sidebar.markdown("## Ask About the Document")
    doc_question = st.sidebar.text_input("Enter your question about the document summary:")
    if doc_question:
        chat_data = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are an AI tutor helping students understand academic material from a document."},
                {"role": "user", "content": f"Document summary:\n{summary}"},
                {"role": "user", "content": doc_question}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        response = requests.post(API_URL, headers=HEADERS, json=chat_data)
        doc_answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "I couldn't find an answer.")
        st.sidebar.markdown("### 💬 Document Q&A Answer:")
        st.sidebar.write(doc_answer)

##############################################
# General Academic Chatbot (Not limited to PDF content)
##############################################
st.markdown("---")
st.subheader("🤖 General Academic Chatbot")
st.markdown("Ask any academic question—even if it's not related to the uploaded document!")

# Voice Input for General Chatbot
st.markdown("### 🎤 Voice Input")
voice_input = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", key="general_voice")
if voice_input:
    st.audio(voice_input["bytes"], format="audio/wav")
    st.write("Processing voice input...")
    # For simplicity, you might later integrate a speech-to-text service here.
    # For now, we will assume the user also types their question.

# Text Input for General Chatbot
general_question = st.text_input("💡 Enter your academic question:", key="general_text_input")
if st.button("Get Answer", key="general_get_answer"):
    if general_question:
        with st.spinner("Thinking... 🤔"):
            general_response = get_detailed_explanation(general_question, student_age, language)
        st.markdown(f"### 📖 Explanation:\n{general_response}", unsafe_allow_html=True)
    else:
        st.warning("Please enter a question!")

##############################################
# Additional Features: OCR for Image-Based Study Material & Feedback
##############################################
st.subheader("📤 Upload Study Material (Image)")
uploaded_image = st.file_uploader("Upload an image (JPEG, PNG) or scanned notes", type=["png", "jpg", "jpeg"])
if uploaded_image:
    with st.spinner("Extracting text..."):
        extracted_text = pytesseract.image_to_string(Image.open(uploaded_image))
    st.text_area("Extracted Text:", extracted_text, height=200)

st.sidebar.markdown("### 💬 Feedback")
feedback = st.sidebar.slider("Rate your experience (1-5)", 1, 5, 3)
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")
    
st.sidebar.markdown("👨‍🏫 Created with ❤️ by Tony")
