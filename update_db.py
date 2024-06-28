import os, argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document

DATA_PATH = 'data'
CHROMA_PATH = 'chroma'

def load_docs(path: str) -> list:
    """
    Load documents from a directory containing PDF files.

    Parameters:
    path (str): The path to the directory containing the PDF files.

    Returns:
    list: A list of Document objects, where each Document represents a PDF file.

    Raises:
    FileNotFoundError: If the specified path does not exist.
    PyPDFError: If there is an error reading a PDF file.
    """
    loader = PyPDFDirectoryLoader(path)
    return loader.load()

def split_docs(docs: list[Document]) -> list[list[str]]:
    """
    Split the given list of Document objects into smaller chunks of text.

    Parameters:
    docs (list[Document]): A list of Document objects, where each Document represents a PDF file.

    Returns:
    list[list[str]]: A list of lists of strings, where each inner list represents a chunk of text extracted from the PDF files.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(docs)

# ollama pull llama3
# ollama serve
def embedding_func():
    """
    Initialize an OllamaEmbeddings object with the specified model.

    Returns:
    OllamaEmbeddings: An initialized OllamaEmbeddings object with the specified model.

    Raises:
    ValueError: If the specified model is not supported.
    """
    embed_f = OllamaEmbeddings(
        model='llama3',
    )
    return embed_f

def chromadb(chunks: list[Document]) -> None:
    """
    Add or update the given list of Document objects in the Chroma database.

    Parameters:
    chunks (list[Document]): A list of Document objects, where each Document represents a chunk of text extracted from the PDF files.

    Returns:
    None: This function does not return any value. It only adds or updates the documents in the database.

    Raises:
    ValueError: If the specified model is not supported.

    The function first initializes a Chroma database with the specified persist directory and embedding function. 
    It then iterates through the chunks, adding an id to the metadata of each chunk. 
    After that, it retrieves the existing items from the database and checks if the chunks already exist in the database. 
    If not, it adds the new chunks to the database.

    Note: This function assumes that the `split_docs` function has already been called to split the PDF files into smaller chunks of text.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_func()
    )

    # Adding an id to metadata of each chunk
    chunk_index = 0
    prev_page_id = None
    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page+1}"

        if current_page_id == prev_page_id:
            chunk_index += 1
        else:
            chunk_index = 0
            prev_page_id = current_page_id
        
        chunk.metadata['id'] = f"{current_page_id}:{chunk_index}"

    
    # Add / update the documents in database
    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    print(f"\n--> Number of existing documents in DB: {len(existing_ids)}")

    # Only add docs that don't already exist
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
    if new_chunk_ids != []:
        db.add_documents(new_chunks, ids=new_chunk_ids)
        print(f"--> Added {len(new_chunk_ids)} documents to DB.\n")
    else:
        print("--> No new documents need to be added.\n")


if __name__ == "__main__":
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("\n--> Clearing Database!")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        else:
            print(f"\n--> The specified directory '{CHROMA_PATH}' does not exist.\n")

    # Create (or update) the data store.
    docs = load_docs(DATA_PATH)
    chunks = split_docs(docs)
    chromadb(chunks)


