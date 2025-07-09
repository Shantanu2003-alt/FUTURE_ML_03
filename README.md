# FUTURE_ML_03
A smart customer support chatbot built using Streamlit and scikit-learn, trained on real-world FAQs.It responds to customer queries, adapts to different tones and personas, and suggests related follow-up questions — all without needing an API key or external service.

https://customersupport-chatbot.streamlit.app/

# Features
100% offline (no OpenAI key required)
Trained on 100 curated customer support Q&A pairs
Persona selector: Support Assistant, Refund Bot, etc.
Tone options: Professional, Friendly, Funny, Minimal
Feedback thumbs (👍 / 👎) for every reply
Smart suggestions for related questions
Analytics dashboard and downloadable chat log

🗂 Folder Structure
project/
│
├── chatbot_data.csv              # Cleaned Q&A dataset
├── customer_support_chatbot.py   # Streamlit app
├── requirements.txt              # Python package requirements
└── README.md                     # You're here

# Installation
1. Clone the repo or upload your files
If cloning:
git clone <your-repo-url>
cd project
2. Install dependencies
pip install -r requirements.txt
3. Run the chatbot

streamlit run customer_support_chatbot.py

# Dataset Overview (chatbot_data.csv)
This file contains 100+ pre-cleaned customer service Q&A pairs with a question_clean column for vectorization.
You can expand or replace it later as needed.

# Output Files
feedback_log.csv — stores user feedback per reply
chat_log.txt — downloadable chat history

# Different Modes and Examples
Persona             Tone                 Example Prompt
Refund Bot          Funny               Where is my refund?
Returns Specialist  Friendly            How do I return my product?
General Help        Minimal             Track my order
Support Assistant   Professional        Cancel my order
