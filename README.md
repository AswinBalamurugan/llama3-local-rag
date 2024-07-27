Please refer: [My Project Collection](https://github.com/AswinBalamurugan/Machine_Learning_Projects/blob/main/README.md)

# Local-RAG: A Retrieval-Augmented Generation Chatbot with Chroma Database and Llama3 Model

## Aim 
The primary aim of this project is to develop a local-RAG chatbot that can efficiently handle large volumes of text data and provide responses to user queries.

## Abstract
This project presents a local-RAG chatbot that utilizes the **Chroma** database and **Llama3** model for efficient information retrieval and response generation. The chatbot is designed to handle large volumes of text data, providing accurate and context-aware responses to user queries. The system is built using Python and leverages the `LangChain` library for prompt construction and chain execution.

## Objectives

1. Implementing a retrieval-augmented generation (RAG) system using the Chroma database and Llama3 model.
2. Developing a user-friendly interface for interacting with the chatbot.
3. Enhancing the chatbot's ability to handle contextual information and provide relevant responses.
4. Providing a comprehensive documentation and README file for easy installation and usage.

## Introduction
- **RAG**
 Retrieval-Augmented Generation is a method that enhances the performance of LLMs by incorporating relevant context from a database of documents.

- **LLM**
 Large Language Models (LLMs) are powerful AI systems that generate human-like text based on the input they receive. They excel at understanding and generating contextual information.

- **Llama3**
 A state-of-the-art LLM developed by Meta AI, Llama3 is a powerful language model that excels at generating human-like text based on the input it receives.

- **Vector Database**
 A vector database is a type of database that uses vector embeddings to represent the data, allowing for efficient and accurate similarity searches. It is designed to efficiently store and retrieve large volumes of text data.

## Method:
The local-RAG chatbot is implemented using the following steps:

1. *Data Preparation*: Collect and preprocess the text data to be used for training the chatbot. This may involve cleaning, tokenizing, and splitting the data into smaller chunks.

2. *Database Creation*: Initialize and populate the Chroma database with the preprocessed text data. Each document in the database is associated with a unique identifier and a vector embedding.

3. *Query Processing*: When a user queries the chatbot, the system performs a maximal marginal relevance search on the Chroma database to retrieve the most relevant documents. The retrieved documents are then used as context for generating the response.

4. *Response Generation*: The context text is passed to a prompt template, which includes the user's question and the retrieved context. The prompt template is then executed using the Llama3 model to generate the response.

5. *User Interface*: Develop a user-friendly interface for interacting with the chatbot. This may include a web application, a command-line interface, or a chatbot framework.

## How to Setup and Use:
To use the local-RAG chatbot, follow these steps:


1. Clone the repository:

```bash
git clone https://github.com/AswinBalamurugan/MLOps_Iris.git
cd iris-classification
```

2. Create a virtual environment and activate it

3. Install the required dependencies: Install the necessary Python packages using the below command. 
```bash
pip install -r requirements.txt
```

4. Download the ollama package: After downloading the ollama package, use the below commands to download and run the required LLM models.
```bash
ollama pull llama3 
ollama pull mxbai-embed-large 
```
The above commands have to be executed only once to download the models. The below command can be executed run the models.
```bash
ollama serve
```

5. Run the chatbot: Start the chatbot server and interact with it using the user interface by running the below command in the same directory where the `rag_chatbot` python file exists.
```bash
streamlit run rag_chatbot.py
```

## Screenshots
![first](https://github.com/AswinBalamurugan/llama3-local-rag/blob/main/images/first.png)
This is how the UI looks when you run the chatbot for the first time.

![doc upload](https://github.com/AswinBalamurugan/llama3-local-rag/blob/main/images/upload_doc.png)
This is how the UI looks when you upload a document and update the database.

![qna](https://github.com/AswinBalamurugan/llama3-local-rag/blob/main/images/sample_qna.png)
This is a sample response from the LLM.

## Conclusion:
The local-RAG chatbot is a powerful and efficient retrieval-augmented generation system that can handle large volumes of text data. By utilizing the Chroma database and Llama3 model, the chatbot provides accurate and context-aware responses to user queries. The system is designed to be easily installed and used, making it an ideal solution for various applications.
