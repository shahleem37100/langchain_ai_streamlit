import streamlit as st
import os
from dotenv import load_dotenv

# LangChain and Ollama imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# --- 1. Configuration and Environment Setup ---
def setup_environment():
    """Loads environment variables and sets up Langsmith tracking."""
    load_dotenv()

    # Langsmith Tracking (for monitoring LLM applications)
    # Note: OPENAI_API_KEY is commented out as it's not directly used by Ollama in this setup.
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

    # Hugging Face tokens (not directly used in this Ollama setup but included for completeness)
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Groq API key (not used in this specific code, but retrieved if available)
    _ = os.getenv("GROQ_API_KEY") # Assign to _ as it's not used further

# Call environment setup at the start of the script
setup_environment()

# --- 2. Prompt Template Definition ---
def define_prompt_template():
    """Defines the chat prompt template for the language model."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user queries accurately and concisely."),
            ("user", "Question:{question}")
        ]
    )

# Initialize the prompt template
chatbot_prompt = define_prompt_template()

# --- 3. Core Response Generation Function ---
def generate_chatbot_response(question: str, model_name: str, temperature: float, max_tokens: int) -> str:
    """
    Generates a response from the Ollama language model using LangChain.

    Args:
        question (str): The user's question.
        model_name (str): The name of the Ollama model to use (e.g., "gemma2:2b").
        temperature (float): Controls the randomness of the model's output (0.0 to 1.0).
        max_tokens (int): Limits the length of the generated response.

    Returns:
        str: The generated answer from the chatbot.
    """
    try:
        # Initialize the Ollama language model
        # Note: temperature and max_tokens are often set at the model level or in generation config,
        # but for simplicity, we'll pass them as parameters to the function, assuming Ollama
        # handles them or they are used for other LLM integrations if this function were more generic.
        # For Ollama, these are typically set when running the model or within LangChain's invoke options.
        # For demonstration, we'll assume the model itself respects these if configured.
        llm = Ollama(model=model_name, temperature=temperature, num_predict=max_tokens)

        # Initialize the output parser to convert LLM output to a string
        output_parser = StrOutputParser()

        # Create a LangChain expression language chain
        # The flow is: Prompt -> LLM -> Output Parser
        chain = chatbot_prompt | llm | output_parser

        # Invoke the chain with the user's question
        answer = chain.invoke({"question": question})
        return answer
    except Exception as e:
        st.error(f"An error occurred during response generation: {e}")
        return "Sorry, I couldn't generate a response at this time."

# --- 4. Streamlit User Interface Layout ---
def setup_streamlit_ui():
    """Sets up the Streamlit application's user interface."""
    st.set_page_config(page_title="Ollama Q&A Chatbot", layout="centered")

    st.title("🤖 Ollama Q&A Chatbot")
    st.markdown("Ask me anything! I'm powered by an Ollama model.")

    # Sidebar for model selection and parameters
    with st.sidebar:
        st.header("Chatbot Settings")

        # Dropdown to select Ollama models (currently fixed for demonstration)
        # The label "Select an Open AI Model" is misleading as it uses Ollama.
        # Changed to "Select an Ollama Model" for clarity.
        selected_engine = st.selectbox(
            "Select an Ollama Model",
            ["gemma2:2b", "llama2", "mistral"], # Example models, ensure these are pulled via Ollama
            index=0 # Default to gemma2:2b
        )

        # Slider for temperature
        selected_temperature = st.slider(
            "Temperature (Randomness)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Controls the creativity of the response. Higher values mean more random output."
        )

        # Slider for max tokens
        selected_max_tokens = st.slider(
            "Max Response Length (Tokens)",
            min_value=50,
            max_value=500, # Increased max to 500 for more flexibility
            value=150,
            step=10,
            help="Sets the maximum number of tokens (words/subwords) in the generated response."
        )
    return selected_engine, selected_temperature, selected_max_tokens

# --- 5. Main Application Logic ---
def main():
    """Main function to run the Streamlit chatbot application."""
    engine, temperature, max_tokens = setup_streamlit_ui()

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input field at the bottom
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_chatbot_response(user_input, engine, temperature, max_tokens)
            st.markdown(response)
        # Add bot message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Initial prompt when no input is given yet
        if not st.session_state.messages:
            st.info("Welcome! Ask me anything to get started.")

if __name__ == "__main__":
    main()