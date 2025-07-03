import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Langsmith Tracking & API Keys ---
# These environment variables are set for Langsmith tracing and API keys.
# Ensure your .env file contains these values:
# OPENAI_API_KEY="your_openai_api_key_here"
# LANGCHAIN_API_KEY="your_langchain_api_key_here"
# LANGCHAIN_TRACING_V2="true"
# LANGCHAIN_PROJECT="Q&A Chatbot with OpenIA"
# HF_TOKEN="your_huggingface_token_here" (currently unused in this specific app)
# HUGGINGFACEHUB_API_TOKEN="your_huggingfacehub_api_token_here" (currently unused)
# GROQ_API_KEY="your_groq_api_key_here" (currently unused)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OpenIA"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY") # Variable loaded but not used in this app

# --- LangChain Components ---

# Prompt Template with memory placeholder for conversational context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        # MessagesPlaceholder will dynamically insert the chat history here
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Question:{question}")
    ]
)

# Cache the LLM instance to prevent re-initialization on every rerun
# This significantly improves performance by only creating the LLM when parameters change.
@st.cache_resource
def get_llm(api_key: str, engine: str, temperature: float, max_tokens: int):
    """
    Initializes and caches the ChatOpenAI language model.
    The LLM is re-initialized only if any of its parameters (api_key, engine, temperature, max_tokens) change.
    """
    # Set the OpenAI API key globally for the openai library
    openai.api_key = api_key
    return ChatOpenAI(model=engine, temperature=temperature, max_tokens=max_tokens)

def generate_response(question: str, chat_history: list, api_key: str, engine: str, temperature: float, max_tokens: int) -> str:
    """
    Generates a response from the OpenAI LLM using LangChain.

    Args:
        question (str): The user's current question.
        chat_history (list): A list of previous messages in the conversation.
                            Each message is a tuple like ("human", "content") or ("ai", "content").
        api_key (str): The OpenAI API key.
        engine (str): The name of the OpenAI model to use (e.g., "gpt-4o").
        temperature (float): The sampling temperature for the LLM.
        max_tokens (int): The maximum number of tokens to generate in the response.

    Returns:
        str: The generated answer from the LLM.
    """
    # Retrieve the cached LLM instance
    llm = get_llm(api_key, engine, temperature, max_tokens)

    # Output parser to convert LLM output to a string
    output_parser = StrOutputParser()

    # Create the LangChain processing chain: Prompt -> LLM -> Output Parser
    chain = prompt | llm | output_parser

    # Invoke the chain with the current question and the full chat history
    answer = chain.invoke({"question": question, "chat_history": chat_history})
    return answer

# --- Streamlit UI ---

st.title("Q&A Chatbot with OpenAI")

# Initialize chat history in Streamlit's session state if it doesn't exist
# This ensures that the chat history persists across reruns of the app.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages from history on app rerun
# Uses st.chat_message for a visually appealing chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Sidebar for Settings ---
st.sidebar.title("Settings")

# Input for OpenAI API Key, masked as a password
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# Dropdown to select OpenAI models
llm_model = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Sliders for LLM parameters: temperature and max_tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=500, value=150, step=10) # Increased max_value for flexibility

# --- Main Chat Interface ---
# Use st.chat_input for a dedicated chat input field at the bottom of the app
user_input = st.chat_input("Go ahead and ask any question...")

# Process user input if available
if user_input:
    # Check if API key is provided before proceeding
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to start chatting.")
    else:
        # Add user message to Streamlit's chat history (for display)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Show a loading spinner while the LLM generates a response
        with st.spinner("Generating response..."):
            # Prepare chat history for LangChain's prompt format
            # LangChain expects a list of tuples: [("human", "message"), ("ai", "message")]
            langchain_chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_chat_history.append(("human", msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_chat_history.append(("ai", msg["content"]))

            # Generate the response using the helper function
            response = generate_response(user_input, langchain_chat_history, api_key, llm_model, temperature, max_tokens)

            # Add assistant message to Streamlit's chat history (for display)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
