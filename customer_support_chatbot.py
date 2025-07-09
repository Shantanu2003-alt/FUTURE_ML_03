# customer_support_chatbot.py

import streamlit as st
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import random

# ------------------- Load Data -------------------
@st.cache_data
def load_data():
    df = pd.read_csv("chatbot_data.csv")
    df = df.dropna(subset=["question_clean", "answer"])
    df = df[df['answer'].str.len() > 15]
    df["question_clean"] = df["question_clean"].astype(str)
    return df

@st.cache_resource
def load_vectorizer(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["question_clean"].astype(str))
    return vectorizer, X

# ------------------- Clean Function -------------------
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\S+|[^a-z0-9\s]", "", text)
    return text.strip()

# ------------------- Chat Memory -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []
if "persona" not in st.session_state:
    st.session_state.persona = "Support Assistant"
if "tone" not in st.session_state:
    st.session_state.tone = "Professional"

# ------------------- Personalities & Tones -------------------
persona_styles = {
    "Support Assistant": "Hi there! How can I help you today?",
    "Returns Specialist": "Hey! Got a return or refund issue? I’m here to help!",
    "Refund Bot": "Hello. Let's talk about refunds. Share your details, please.",
    "General Help": "Hi! Ask me anything, I’ll try to assist."
}

tone_styles = {
    "Professional": lambda msg: msg,
    "Friendly": lambda msg: f"😊 Sure! {msg}",
    "Funny": lambda msg: f"😄 You got it! {msg} (and maybe a joke later!)",
    "Minimal": lambda msg: msg.split('.')[0] + '.' if '.' in msg else msg
}

# ------------------- App Layout -------------------
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")
st.markdown("""
    <style>
    .bubble {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #e6f7ff;
    }
    .bot-bubble {
        background-color: #fff2e6;
    }
    .suggestion {
        background-color: #f0f0f0;
        border: none;
        padding: 8px 15px;
        margin: 4px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Customer Support Chatbot")
st.caption("Trained on 100 real-world support Q&A pairs. Enjoy multiple personas, tones, and smarter logic.")

# Persona and Tone selection
persona = st.selectbox("Choose a persona:", list(persona_styles.keys()))
st.session_state.persona = persona
st.info(persona_styles[persona])

tone = st.selectbox("Choose a tone:", list(tone_styles.keys()))
st.session_state.tone = tone

# Load data and vectorizer
qa_df = load_data()
vectorizer, X = load_vectorizer(qa_df)

# ------------------- Helper: Related Suggestions -------------------
def get_suggestions(user_input, top_k=3):
    cleaned = clean(user_input)
    user_vec = vectorizer.transform([cleaned])
    sims = cosine_similarity(user_vec, X).flatten()
    top_indices = sims.argsort()[-top_k-1:-1][::-1]
    return qa_df.iloc[top_indices]["question"].tolist()

# ------------------- Bot Response -------------------
def get_bot_response(user_input):
    cleaned = clean(user_input)
    if not cleaned:
        return tone_styles[st.session_state.tone]("Can you please enter a valid message?"), []
    user_vec = vectorizer.transform([cleaned])
    sims = cosine_similarity(user_vec, X)
    top_idx = sims.argmax()
    confidence = sims[0, top_idx]

    if confidence < 0.4:
        return tone_styles[st.session_state.tone]("Sorry, I didn’t understand that. You can ask about orders, returns, or shipping!"), []

    response = qa_df.iloc[top_idx]['answer']
    suggestions = get_suggestions(user_input)
    return tone_styles[st.session_state.tone](str(response)), suggestions

# ------------------- Chat Interface -------------------
with st.form("chat_form", clear_on_submit=True):
    user_message = st.text_input("You:", placeholder="Ask your question here...", key="input")
    submitted = st.form_submit_button("Send")

if submitted and user_message:
    st.session_state.chat_history.append(("user", user_message))

    if re.search(r"order \d+", user_message.lower()):
        order_id = re.search(r"order (\d+)", user_message.lower()).group(1)
        bot_reply = f"Your order {order_id} is out for delivery and will arrive by tomorrow!"
        suggestions = []
    else:
        bot_reply, suggestions = get_bot_response(user_message)

    st.session_state.chat_history.append(("bot", bot_reply))
    st.session_state.chat_history.append(("suggest", suggestions))

# ------------------- Chat Display -------------------
st.markdown("---")
st.subheader("💬 Chat")
for i, entry in enumerate(st.session_state.chat_history):
    if entry[0] == "user":
        st.markdown(f"<div class='bubble user-bubble'><strong>You:</strong> {entry[1]}</div>", unsafe_allow_html=True)
    elif entry[0] == "bot":
        st.markdown(f"<div class='bubble bot-bubble'><strong>🤖 {st.session_state.persona}:</strong> {entry[1]}</div>", unsafe_allow_html=True)
        # Feedback buttons
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            if st.button(f"👍 Helpful", key=f"like_{i}"):
                st.session_state.feedback.append((entry[1], "👍"))
        with feedback_col2:
            if st.button(f"👎 Not Helpful", key=f"dislike_{i}"):
                st.session_state.feedback.append((entry[1], "👎"))
    elif entry[0] == "suggest" and entry[1]:
        st.markdown("**Related Questions:**")
        for q in entry[1]:
            st.markdown(f"<button class='suggestion'>{q}</button>", unsafe_allow_html=True)

# ------------------- Analytics -------------------
st.markdown("---")
st.subheader("📊 Analytics")
st.write("Queries handled:", len([m for m in st.session_state.chat_history if m[0]=='user']))
st.write("Feedback received:", len(st.session_state.feedback))

# ------------------- Download Chat Log -------------------
if st.button("📥 Download Chat Log"):
    chat_log = "\n".join([f"You: {u}\nBot: {b}" for u, b in zip(
        [m[1] for m in st.session_state.chat_history if m[0]=="user"],
        [m[1] for m in st.session_state.chat_history if m[0]=="bot"]
    )])
    st.download_button("Download as .txt", chat_log, file_name="chat_log.txt")

# ------------------- Optional: Save Feedback CSV -------------------
if len(st.session_state.feedback) > 0:
    df_feedback = pd.DataFrame(st.session_state.feedback, columns=["Response", "Feedback"])
    df_feedback.to_csv("feedback_log.csv", index=False)

