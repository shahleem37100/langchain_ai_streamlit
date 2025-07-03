import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# --- Configuration and Constants ---
# Define the LLM model to be used for summarization
LLM_MODEL = "gemma2-9b-it"
# Define the prompt template for the summarization task
SUMMARIZE_PROMPT_TEMPLATE = """
Provide a concise summary of the following content in approximately 300 words.
Focus on the main ideas and key takeaways.

Content:
{text}
"""

# --- Helper Functions ---

def load_content(url: str):
    """
    Loads content from a given URL, supporting YouTube and general web pages.

    Args:
        url (str): The URL of the content to load.

    Returns:
        list: A list of LangChain Document objects containing the loaded content.
    """
    st.info(f"Loading content from: {url}")
    if "youtube.com" in url:
        # Load transcript from YouTube URL
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    else:
        # Load content from general web page URL
        # ssl_verify=False is used here for broader compatibility but can be a security risk.
        # Consider setting up proper SSL certificate handling in a production environment.
        loader = UnstructuredURLLoader(
            urls=[url],
            ssl_verify=False,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
        )
    return loader.load()

def run_summarization_chain(llm, docs):
    """
    Runs the LangChain summarization chain on the provided documents.

    Args:
        llm: The initialized LangChain LLM instance (e.g., ChatGroq).
        docs (list): A list of LangChain Document objects.

    Returns:
        str: The generated summary.
    """
    # Create a PromptTemplate instance from the defined template
    prompt = PromptTemplate(template=SUMMARIZE_PROMPT_TEMPLATE, input_variables=["text"])
    
    # Load the summarization chain.
    # "stuff" chain type puts all document content into a single prompt.
    # This is suitable for documents that fit within the LLM's context window.
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    
    # Run the chain with the loaded documents
    return chain.run(docs)

# --- Streamlit Application Layout and Logic ---

def main():
    """
    Main function to run the Streamlit application.
    """
    # Set Streamlit page configuration
    st.set_page_config(page_title="Langchain: URL Content Summarizer", layout="centered")
    
    # Application header
    st.title("Langchain: Summarize Content from Website")
    st.subheader('Enter a URL to get a concise summary')

    # Sidebar for API Key input
    with st.sidebar:
        groq_api_key = st.text_input("Groq API Key", value="", type="password")
        st.markdown("Get your Groq API key from [Groq Console](https://console.groq.com/keys)")

    # Main input for URL
    generic_url = st.text_input("URL to Summarize", placeholder="e.g., https://www.example.com/article or https://www.youtube.com/watch?v=video_id")

    # Summarize button
    if st.button("Summarize Content"):
        # --- Input Validation ---
        if not groq_api_key.strip():
            st.error("Please provide your Groq API Key.")
            return
        if not generic_url.strip():
            st.error("Please enter a URL to summarize.")
            return
        if not validators.url(generic_url):
            st.error("Please enter a valid URL.")
            return

        try:
            # Initialize the LLM with the provided API key and model
            llm = ChatGroq(model=LLM_MODEL, groq_api_key=groq_api_key)

            with st.spinner("Summarizing content... This might take a moment."):
                # Load content from the URL
                docs = load_content(generic_url)
                
                # Run the summarization process
                output_summary = run_summarization_chain(llm, docs)

                # Display the summary
                st.success("Summary:")
                st.write(output_summary)

        except Exception as e:
            # Catch and display any errors during the process
            st.error(f"An error occurred: {e}")
            st.exception(e) # Display full exception details for debugging

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
