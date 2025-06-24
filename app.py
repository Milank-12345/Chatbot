import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="💬 Loan Advisory Assistant", layout="centered", page_icon="💸")

# App Header
st.markdown("""
    <div style='text-align: center; margin-top: -40px;'>
        <h1 style='color: #4A90E2;'>💬 Loan Advisory Assistant</h1>
        <p style='color: #888;'>Smart, friendly, and tailored loan guidance.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
        This chatbot helps you:
        - Understand loan types
        - Choose the best loan
        - Estimate repayment terms
        - Get step-by-step guidance

        Powered by:
        - 🌐 **Google Gemini Flash**
        - 🦜 **LangChain**
        - 🚀 **Streamlit**
    """)
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.success("Chat history cleared!")

# API key loading
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ `GOOGLE_API_KEY` not found in .env file.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# Prompt template
prompt_template = """
You are a friendly and professional loan advisory assistant. Your goal is to help users find the best loan options by asking relevant questions and providing tailored advice. Use a conversational tone, keep answers short and to the point, and guide the user step-by-step.

Start by asking the user's name if not already provided. Make the user feel confident that you're the best assistant to help them.

Current conversation history:
{chat_history}

User's latest input: {user_input}

Based on the conversation, ask the next relevant question to understand the user's loan needs (e.g., loan type, amount, purpose, credit score, income, or repayment period), or provide tailored advice if you have enough information. 

If starting the conversation, begin by asking: "Hi there! May I know your name so I can assist you better?"

Response:
"""

# Setup session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

if "chain" not in st.session_state:
    prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=prompt_template)
    st.session_state.chain = LLMChain(llm=llm, prompt=prompt, memory=st.session_state.memory)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# First-time greeting
if not st.session_state.chat_history:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("👋 Hello! I'm your personal loan advisor. Let's find the best loan for your needs.")

# Show chat history
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**You:** {user_msg}")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"**Advisor:** {bot_msg}")

# User input and LLM response
user_input = st.chat_input("Type your loan-related question...")

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**You:** {user_input}")

    with st.spinner("🤔 Thinking..."):
        response = st.session_state.chain.run(user_input=user_input)

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(f"**Advisor:** {response}")

    st.session_state.chat_history.append((user_input, response))



# # prompt_template_v2 = """
# # You are a friendly and professional loan advisory assistant. Your job is to help users find the best loan options by asking just enough relevant questions, then giving clear, personalized recommendations. Use a warm, conversational tone. Keep responses short and focused.

# # Only ask follow-up questions if the information is incomplete. Once you have enough details (e.g., loan type, amount, purpose, credit score, income, and repayment period), stop asking and provide a recommendation.

# # Conversation history:
# # {chat_history}

# # User's latest input: {user_input}

# # If this is the first message, begin with:
# # "Hi there! May I know your name so I can assist you better?"

# # Otherwise, ask the next most relevant question or give a tailored recommendation based on what you know so far.

# # Response:
# # """
