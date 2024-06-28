import argparse
from update_db import embedding_func, CHROMA_PATH, Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embed_func = embedding_func()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_func)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    sources = '\n'.join(['| '.join(source.split(':')[:2]) for source in sources])
    formatted_response = f"\n--> Response: \n{response_text}\n----------\n\n--> Sources (doc| page number):\n{sources}\n"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)



