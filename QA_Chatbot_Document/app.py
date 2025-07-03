import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Not explicitly used in the original RAG chain, but good to keep if parsing is needed
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_core.documents import Document # Not directly used in the main flow, but useful for type hinting

# --- Configuration Constants ---
# It's good practice to define constants at the top for easy modification.
PDF_DIRECTORY = "research_papers"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LLM_MODEL_NAME = "gemma2-9b-it"
MAX_DOCUMENTS_TO_PROCESS = 50 # Limiting to 50 as in the original code

class RAGQueryApp:
    """
    A Streamlit application for Question Answering using Retrieval Augmented Generation (RAG).
    It leverages LangChain for document processing, vector storage, and chain creation,
    and Groq for the Language Model.
    """

    def __init__(self):
        """
        Initializes the RAG application, loading environment variables and setting up
        the Streamlit page configuration.
        """
        self._load_environment_variables()
        self._set_streamlit_page_config()
        self.llm = None
        self.prompt = None

    def _load_environment_variables(self):
        """
        Loads environment variables from a .env file and sets them as OS environment variables.
        """
        load_dotenv()
        # Ensure all necessary API keys are loaded
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true" # Enable LangChain tracing
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

        # Basic check for essential keys
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found in environment variables. Please set it.")
            st.stop()
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not found in environment variables. Please set it.")
            st.stop()

    def _set_streamlit_page_config(self):
        """
        Sets the basic configuration for the Streamlit page.
        """
        st.set_page_config(
            page_title="RAG Document Q&A with Groq",
            page_icon="📚",
            layout="centered",
            initial_sidebar_state="auto"
        )
        st.title("📚 RAG Document Q&A With Groq")
        st.markdown(
            """
            This application allows you to query your research papers (PDFs in the `research_papers` folder)
            using a Retrieval Augmented Generation (RAG) system powered by Groq's LLM.
            """
        )

    def _initialize_llm_and_prompt(self):
        """
        Initializes the ChatGroq LLM and the ChatPromptTemplate.
        This is done lazily or once per session.
        """
        if self.llm is None:
            try:
                groq_api_key = os.getenv("GROQ_API_KEY")
                self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=LLM_MODEL_NAME)
                self.prompt = ChatPromptTemplate.from_template(
                    """
                    Answer the question based on the provided context only.
                    Please provide only the most accurate response based on the question.
                    If the answer is not available in the context, state that you cannot answer.

                    <context>
                    {context}
                    </context>

                    Question: {input}
                    """
                )
            except Exception as e:
                st.error(f"Error initializing LLM or prompt: {e}")
                st.stop()

    def create_vectors_embedding(self):
        """
        Loads PDF documents, splits them into chunks, creates embeddings,
        and stores them in a FAISS vector database.
        This process is cached in Streamlit's session state.
        """
        if "vectors" not in st.session_state:
            with st.spinner("Loading documents and creating vector embeddings... This may take a moment."):
                try:
                    st.session_state.embeddings = OpenAIEmbeddings()
                    # Ensure the PDF directory exists
                    if not os.path.exists(PDF_DIRECTORY):
                        st.error(f"The directory '{PDF_DIRECTORY}' does not exist. Please create it and place your PDF research papers inside.")
                        return False

                    st.session_state.loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
                    docs = st.session_state.loader.load()

                    if not docs:
                        st.warning(f"No PDF documents found in '{PDF_DIRECTORY}'. Please add some PDFs to proceed.")
                        return False

                    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                    )
                    # Limit the number of documents processed as per original code
                    final_documents = st.session_state.text_splitter.split_documents(
                        docs[:MAX_DOCUMENTS_TO_PROCESS]
                    )

                    st.session_state.vectors = FAISS.from_documents(
                        final_documents, st.session_state.embeddings
                    )
                    st.success("Vector Database is ready! You can now ask questions.")
                    return True
                except Exception as e:
                    st.error(f"An error occurred during document embedding: {e}")
                    st.info("Please ensure you have valid PDF files in the 'research_papers' directory and your OpenAI API key is correctly set.")
                    return False
        else:
            st.info("Vector Database is already loaded.")
            return True


    def run_rag_query(self, user_query: str):
        """
        Executes the RAG query process: retrieves relevant documents,
        and uses the LLM to generate an answer based on the context.

        Args:
            user_query (str): The question asked by the user.
        """
        if "vectors" not in st.session_state:
            st.warning("Please click 'Load Documents & Create Embeddings' first.")
            return

        # Ensure LLM and prompt are initialized
        self._initialize_llm_and_prompt()
        if self.llm is None or self.prompt is None:
            st.error("LLM or Prompt could not be initialized. Cannot run query.")
            return

        with st.spinner("Getting your answer..."):
            try:
                document_chain = create_stuff_documents_chain(self.llm, self.prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_query})
                end_time = time.process_time()

                st.write(f"**Answer:** {response['answer']}")
                st.info(f"Response generated in: {end_time - start_time:.2f} seconds")

                with st.expander("🔍 Document Similarity Search (Context Used)"):
                    if 'context' in response and response['context']:
                        for i, doc in enumerate(response['context']):
                            st.markdown(f"**Document Chunk {i+1}:**")
                            st.write(doc.page_content)
                            st.markdown("---")
                    else:
                        st.info("No specific context documents were retrieved for this query.")

            except Exception as e:
                st.error(f"An error occurred during the RAG query: {e}")
                st.info("Please check your query or ensure the vector database is correctly loaded.")

    def _setup_streamlit_ui(self):
        """
        Sets up the main Streamlit UI elements for the application.
        """
        # Button to trigger document embedding
        if st.button("Load Documents & Create Embeddings"):
            self.create_vectors_embedding()

        # Input for user query
        user_prompt = st.text_input(
            "Enter your query from the research paper:",
            placeholder="e.g., What are the key findings of the paper on quantum computing?"
        )

        # Run query if user_prompt is provided
        if user_prompt:
            self.run_rag_query(user_prompt)

# --- Main execution block ---
if __name__ == "__main__":
    app = RAGQueryApp()
    app._setup_streamlit_ui()