## Conversational RAG with PDF Uploads and Chat History (Customized Version)
# This script creates an interactive Streamlit web application that enables
# users to upload PDF documents and engage in a conversational Q&A session
# with an AI assistant. The core functionality is powered by Retrieval-Augmented
# Generation (RAG) using LangChain, Groq for the Large Language Model (LLM),
# and ChromaDB as the vector store. Crucially, the application maintains
# chat history across turns, allowing the AI to understand context from
# previous interactions.

# --- 1. Library Imports ---
# Standard Library Imports
import os
import tempfile # Essential for safely handling temporary files

# Third-Party Library Imports
import streamlit as st
from dotenv import load_dotenv

# LangChain Specific Imports
# Chains for orchestrating LLM calls and retrieval
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory

# Vector Store for document indexing and retrieval
from langchain_chroma import Chroma

# Chat Message History for managing conversation context
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Prompt Templates for guiding the LLM's behavior
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Language Model (LLM) Integration
from langchain_groq import ChatGroq

# Embedding Model for converting text to numerical vectors
from langchain_huggingface import HuggingFaceEmbeddings

# Document Loaders and Text Splitters for data preparation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# --- 2. Configuration and Global Initialization ---

# Load environment variables from a .env file.
# This is a best practice for securely managing API keys and sensitive credentials,
# keeping them out of the main codebase and preventing accidental exposure.
load_dotenv()

# Retrieve Hugging Face token from environment variables.
# This token is often required for certain Hugging Face models, especially commercial ones
# or for higher rate limits.
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize the embedding model.
# 'all-MiniLM-L6-v2' is a popular, efficient, and good-performing Sentence-BERT model
# suitable for generating embeddings for semantic search. It transforms text chunks
# into numerical vectors that can be compared for similarity.
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading embedding model: {e}. Please ensure 'HF_TOKEN' is set correctly and the model name is valid.")
    st.stop() # Stop the app if embeddings can't be loaded, as they are fundamental.


# --- 3. Streamlit Application User Interface Setup ---

st.set_page_config(page_title="Conversational RAG with PDFs", layout="wide")
st.title("📚 Conversational RAG with PDF Uploads and Chat History")
st.write(
    """
    Upload your PDF documents and engage in a dynamic conversation with an AI assistant.
    The assistant leverages Retrieval-Augmented Generation (RAG) to answer your questions
    based on the content of your uploaded PDFs, intelligently remembering the context
    of your ongoing chat.
    """
)

# Input for the Groq API Key.
# It's crucial for the user to provide their API key. Using `type="password"`
# prevents the key from being displayed in plain text on the screen.
groq_api_key = st.text_input("🔑 Enter your Groq API key:", type="password")


# --- 4. Helper Functions for RAG Pipeline ---

def get_chat_session_history(session_identifier: str) -> BaseChatMessageHistory:
    """
    Manages and retrieves the chat history for a specific session.
    This function ensures that each unique session (identified by `session_identifier`)
    has its own persistent chat history stored within Streamlit's session state.

    Args:
        session_identifier (str): A unique string identifier for the current chat session.

    Returns:
        BaseChatMessageHistory: An instance of ChatMessageHistory for the specified session.
    """
    # Streamlit's session_state is used for stateful management across reruns.
    # If the main 'chat_history_store' dictionary doesn't exist, initialize it.
    if 'chat_history_store' not in st.session_state:
        st.session_state.chat_history_store = {}
        # Personal note: Consider adding a max history size or disk persistence for very long chats.

    # If the specific session_identifier's history doesn't exist, create it.
    if session_identifier not in st.session_state.chat_history_store:
        st.session_state.chat_history_store[session_identifier] = ChatMessageHistory()
        # Thought process: This allows multiple users or separate conversations
        # to exist concurrently without mixing their histories.
    return st.session_state.chat_history_store[session_identifier]

def load_and_split_documents(uploaded_files_list) -> list:
    """
    Loads content from uploaded PDF files, converts them into LangChain Document objects,
    and then splits these documents into smaller, manageable chunks.

    Args:
        uploaded_files_list (list): A list of Streamlit UploadedFile objects.

    Returns:
        list: A list of LangChain Document objects, representing the text chunks.
    """
    all_raw_documents = []
    # Iterating through each uploaded file to process them individually.
    for uploaded_file in uploaded_files_list:
        # Use tempfile to create a secure temporary file.
        # This handles unique filenames automatically and ensures proper cleanup.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # PyPDFLoader is used to extract text content page by page from the PDF.
            loader = PyPDFLoader(temp_file_path)
            docs_from_current_pdf = loader.load()
            all_raw_documents.extend(docs_from_current_pdf)
            # Personal note: For very large PDFs, consider lazy loading or batch processing.
        finally:
            # Ensure the temporary file is removed after processing, regardless of success or failure.
            os.remove(temp_file_path)

    # Initialize RecursiveCharacterTextSplitter for chunking.
    # 'chunk_size' determines the maximum size of each text segment.
    # 'chunk_overlap' ensures continuity between chunks, which can help in
    # retaining context when a piece of information spans across two chunks.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    document_chunks = text_splitter.split_documents(all_raw_documents)
    return document_chunks

def initialize_vector_store(document_chunks: list, embeddings_model: HuggingFaceEmbeddings) -> Chroma:
    """
    Initializes a Chroma vector store with the provided document chunks and embeddings model.
    This creates an index of the document content, allowing for efficient semantic search.

    Args:
        document_chunks (list): A list of LangChain Document objects (text chunks).
        embeddings_model (HuggingFaceEmbeddings): The model used to generate embeddings.

    Returns:
        Chroma: An initialized Chroma vector store.
    """
    # Chroma.from_documents takes text chunks and their embeddings, and builds a searchable index.
    vector_store = Chroma.from_documents(documents=document_chunks, embedding=embeddings_model)
    return vector_store

def create_rag_pipeline(llm_model: ChatGroq, document_retriever) -> RunnableWithMessageHistory:
    """
    Constructs the complete conversational RAG chain using LangChain components.
    This involves setting up the contextualization, retrieval, and answer generation steps.

    Args:
        llm_model (ChatGroq): The initialized Groq Language Model instance.
        document_retriever: The retriever object (e.g., from a Chroma vector store).

    Returns:
        RunnableWithMessageHistory: The complete RAG chain, capable of handling
                                    conversation history.
    """
    # --- Step 1: Contextualize the user's question with chat history ---
    # This prompt is crucial for multi-turn conversations. It guides the LLM
    # to rephrase the current user query into a standalone question that
    # incorporates context from previous turns, making it independent of history
    # for the retrieval step.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, which might "
        "reference context from the chat history, formulate a standalone "
        "question that can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and "
        "otherwise return it as is. This ensures the retriever gets an "
        "unambiguous query for relevant document chunks."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # Placeholder for the actual chat history messages.
            ("human", "{input}"), # The current user's input.
        ]
    )
    # The history-aware retriever first processes the user's input with history
    # to get a clearer query, then uses that query to fetch documents.
    history_aware_retriever = create_history_aware_retriever(
        llm_model, document_retriever, contextualize_q_prompt
    )

    # --- Step 2: Answer the question using retrieved context ---
    # This prompt instructs the LLM on how to generate an answer based on
    # the retrieved documents. It emphasizes conciseness and grounding
    # the answer strictly in the provided context to minimize hallucinations.
    qa_system_prompt = (
        "You are an intelligent assistant specializing in answering questions "
        "based on provided context. Use the following pieces of retrieved context "
        "to answer the question accurately and concisely. "
        "If the answer is not found within the provided context, clearly state "
        "that you don't know the answer. Aim for a maximum of three sentences "
        "and keep the answer highly relevant to the question and context."
        "\n\n"
        "Retrieved Context:\n{context}" # The placeholder where retrieved document content will be inserted.
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"), # Pass chat history for conversational flow (optional for answer, but good for context)
            ("human", "{input}"), # The refined user question.
        ]
    )
    # This chain combines the LLM with the context and the prompt to generate the final answer.
    Youtube_chain = create_stuff_documents_chain(llm_model, qa_prompt)

    # --- Step 3: Combine retrieval and answer generation ---
    # This creates the main RAG chain that orchestrates the flow:
    # 1. User query comes in.
    # 2. History-aware retriever processes it and fetches relevant documents.
    # 3. Question-answer chain takes documents and refined query to generate response.
    full_rag_chain = create_retrieval_chain(
        history_aware_retriever, Youtube_chain
    )

    # --- Step 4: Integrate with Streamlit's session history ---
    # RunnableWithMessageHistory is a powerful LangChain feature that automatically
    # manages the `ChatMessageHistory` object. It handles both reading history
    # before invocation and updating it after.
    conversational_rag_chain = RunnableWithMessageHistory(
        full_rag_chain,
        get_chat_session_history, # Our custom function to get/create history for a session.
        input_messages_key="input", # Key in the input dictionary that holds the current user message.
        history_messages_key="chat_history", # Key in the prompt that expects the chat history.
        output_messages_key="answer" # Key in the output dictionary that holds the AI's answer.
        # Thought process: Explicitly naming keys improves readability and debuggability.
    )
    return conversational_rag_chain


# --- 5. Main Application Execution Block ---

# Check if the Groq API key is provided before proceeding with LLM initialization.
if groq_api_key:
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
        # Personal note: Consider adding a model selection dropdown for more flexibility.
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}. Please check your API key.")
        st.stop() # Stop the app if the LLM cannot be initialized.

    # User input for current chat session ID. This allows separate conversations.
    current_chat_session_id = st.text_input(
        "📝 Enter a Session ID (e.g., 'my_research_chat'):", value="default_rag_session"
    )

    # PDF file uploader. Allows multiple files.
    uploaded_pdf_files = st.file_uploader(
        "⬆️ Upload PDF document(s)", type="pdf", accept_multiple_files=True
    )

    # Process uploaded files and set up RAG components only if files are present.
    if uploaded_pdf_files:
        st.info("🔄 Processing PDFs and building knowledge base... This may take a moment.")
        with st.spinner("Loading and embedding documents..."):
            document_chunks = load_and_split_documents(uploaded_pdf_files)

            if not document_chunks:
                st.warning("No text extracted from PDFs. Please ensure PDFs are not image-only.")
                st.stop() # Stop if no content can be processed.

            # Initialize the vector store and retriever.
            # This is where the magic of semantic search begins.
            vector_store_instance = initialize_vector_store(document_chunks, embeddings_model)
            document_retriever = vector_store_instance.as_retriever()

            # Set up the complete conversational RAG pipeline.
            # This is the brain of our Q&A system.
            conversational_rag_pipeline = create_rag_pipeline(llm, document_retriever)
        st.success("✅ PDFs processed and RAG system is ready to chat!")

        # User input for their question.
        user_question_input = st.text_input("💬 Ask your question about the documents:")

        if user_question_input:
            # Retrieve the specific session's history before invoking the chain.
            current_session_history = get_chat_session_history(current_chat_session_id)

            with st.spinner("Thinking and retrieving information..."):
                # Invoke the RAG pipeline. The `config` parameter ensures the
                # correct session history is used by `RunnableWithMessageHistory`.
                rag_response = conversational_rag_pipeline.invoke(
                    {"input": user_question_input},
                    config={"configurable": {"session_id": current_chat_session_id}},
                )

            # Display the AI assistant's answer.
            st.markdown(f"**🤖 Assistant:** {rag_response['answer']}")

            # Display the full chat history in a more readable format.
            st.subheader("Conversation History:")
            # Iterating through `messages` allows us to display human and AI messages distinctively.
            for message in current_session_history.messages:
                if message.type == "human":
                    st.text(f"You: {message.content}")
                elif message.type == "ai":
                    st.text(f"Assistant: {message.content}")

    else:
        st.info("⬆️ Upload PDF files above to enable the chat functionality.")

else:
    st.warning("⚠️ Please enter your Groq API Key to activate the AI assistant.")
    # Personal note: Maybe provide a link to Groq for key generation here.