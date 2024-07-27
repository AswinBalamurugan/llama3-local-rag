from io import FileIO
import os, subprocess, time
import pandas as pd
from update_db import embedding_func, CHROMA_PATH, DATA_PATH, Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
import streamlit as st

PROMPT_TEMPLATE = [
    ("system", """You are a helpful assistant. Please respond to the questions. 
                If the context is irrelevant, ignore it."""),
    ("user", "Context: {context}\n---\nQuestion: {question}")
]

def query_rag(query_text: str) -> tuple[str,pd.DataFrame]:
    """
    This function performs a retrieval-augmented generation (RAG) query using the Chroma database and Llama3 model.

    Parameters:
    query_text (str): The input query text.

    Returns:
    tuple: A tuple containing the generated response and a DataFrame of sources.

    Raises:
    None

    Note:
    This function uses the Chroma database for document retrieval, the Llama3 model for response generation,
    and the LangChain library for prompt construction and chain execution.
    """

    # Initialize the embedding function and Chroma database
    embed_func = embedding_func()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_func)

    # Perform maximal marginal relevance search on the Chroma database
    results = db.max_marginal_relevance_search(
        query_text,
        k = 5,
        fetch_k = 10,
        lambda_multiplier = 0.5
    )

    # Construct the context text from the retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])

    # Initialize the prompt template and Llama3 model
    prompt_template = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)
    model = Ollama(model="llama3")

    # Construct the chain and generate the response
    chain = prompt_template | model
    response = chain.invoke({"context": context_text, "question": query_text})

    # Extract and format the sources from the retrieved documents
    sources = [doc.metadata.get("ids") for doc in results]
    source_data = [source.split(':')[:2] for source in sources if source]
    df_sources = pd.DataFrame(source_data, columns=['Document', 'Page Number'])

    # Return the generated response and sources
    return response, df_sources

def save_uploaded_file(uploaded_file: FileIO)->bool:
    """
    This function saves an uploaded file to the 'data' directory.

    Parameters:
    uploaded_file (FileIO): The uploaded file object. It should have attributes like name and getbuffer().

    Returns:
    bool: True if the file is saved successfully, False otherwise.

    Raises:
    None

    Note:
    The function uses the os.path.join() method to construct the file path.
    It opens the file in binary write mode ('wb') and writes the file content using the getbuffer() method.
    If any error occurs during the file operation, it catches the exception and returns False.
    """
    os.makedirs(DATA_PATH, exist_ok=True)
    try:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('LLAMA 3 model Chatbot')

# File uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        if save_uploaded_file(uploaded_file):
            st.success(f"File {uploaded_file.name} is ready to be uploaded to database!", icon="✅")
        else:
            st.error(f"Error saving file {uploaded_file.name}")


if st.button("Update Database"):
    if os.path.exists(DATA_PATH):
        subprocess.run(["python", "update_db.py", "--add"], check=True)
        st.success("Database updated!", icon="✅")
    else:
        st.error("No files uploaded.", icon='🚨')


# Initialize the embedding function and Chroma database
embed_func = embedding_func()
if os.path.exists(CHROMA_PATH) and os.path.exists(DATA_PATH):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_func)

    # List the documents available in the database
    results = db.get()
    names = set([meta['source'] for meta in results['metadatas']])
    if names:
        st.write("Documents available in the database:")
        for i,name in enumerate(names):
            st.write(f'{i+1}. {name}')
    else:
        st.warning('The database is empty!', icon="⚠️")

    # Dropdown to select whether to clear the database
    option = st.selectbox("Which files do you want to remove from the database?", \
                        ["Select an option", "All"]+[name for name in names], index=0)

    if option == "All":
        if os.path.exists(DATA_PATH):
            subprocess.run(["python", "update_db.py", "--clear"], check=True)
            st.success("Database cleared! Page will refresh in 5 seconds.", icon="✅")
            time.sleep(5)
            st.rerun()
        else:
            st.error("Could not clear the database!", icon="🚨")
    elif option in names:
        if os.path.exists(DATA_PATH):
            subprocess.run(["python", "update_db.py", "--remove", option], check=True)
            st.success(f"Removed document {option}! Page will refresh in 5 seconds.", icon="✅")
            time.sleep(5)
            st.rerun()
        else:
            st.error(f"Could not clear the database for document {option}!", icon="���")
    elif option == "No":
        st.info("Database clearing canceled.", icon="🚫")
else:
    st.warning('The database is empty!', icon="⚠️")


# Display chat history
for i, (query, response, sources) in enumerate(st.session_state.chat_history):
    st.markdown(f'### Query {i+1}:')
    st.write(query)

    st.markdown(f"### Response {i+1}:")
    st.write(response)
    
    st.markdown(f"### Sources for Query {i+1}:")
    st.table(sources)
    
    st.markdown("---")

# Input for new query
input_text = st.text_input("Ask your question!")

if input_text:
    response, sources = query_rag(input_text)
    
    # Add new interaction to chat history
    st.session_state.chat_history.append((input_text, response, sources))
    
    # Display new response
    st.markdown('### Query:')
    st.write(input_text)

    st.markdown("### Response:")
    st.write(response)
    
    st.markdown("### Sources:")
    st.table(sources)
    
    st.markdown("---")

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()