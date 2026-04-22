import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================
# Config
# =========================

DB_FAISS_PATH = "vectorstore/db_faiss"


# =========================
# Load Vector DB
# =========================

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db


# =========================
# Prompt
# =========================

CUSTOM_PROMPT_TEMPLATE = """
Use the context below to answer the question.
If you don't know, say you don't know.

Context:
{context}

Question:
{input}

Answer:
"""


def get_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "input"]
    )


# =========================
# Build Chain (MODERN)
# =========================

def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # ✅ stable Groq model
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = get_prompt()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# =========================
# Streamlit UI
# =========================

def main():
    st.title("💬 Medical Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    user_input = st.chat_input("Ask your question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            vectorstore = get_vectorstore()
            chain = build_chain(vectorstore)

            response = chain.invoke(user_input)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()