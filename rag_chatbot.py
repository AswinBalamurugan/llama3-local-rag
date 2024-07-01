import pandas as pd
from update_db import embedding_func, CHROMA_PATH, Chroma
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
import streamlit as st

PROMPT_TEMPLATE = [
    ("system", """You are a helpful assistant. Please respond to the questions. 
                If the context is irrelevant, ignore it and mention that is context is irrelevant."""),
    ("user", "Context: {context}\n---\nQuestion: {question}")
]

def query_rag(query_text: str):
    embed_func = embedding_func()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embed_func)

    results = db.max_marginal_relevance_search(
        query_text,
        k = 5,
        fetch_k = 10,
        lambda_multiplier = 0.5
    )

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_messages(PROMPT_TEMPLATE)

    model = Ollama(model="llama3")
    chain = prompt_template | model
    response = chain.invoke({"context": context_text, "question": query_text})

    sources = [doc.metadata.get("id", None) for doc in results]
    source_data = [source.split(':')[:2] for source in sources if source]
    df_sources = pd.DataFrame(source_data, columns=['Document', 'Page Number'])

    return response, df_sources

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title('LLAMA 3 model Chatbot')

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
    st.experimental_rerun()