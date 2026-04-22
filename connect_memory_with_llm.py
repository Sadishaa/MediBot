import os

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS


# =========================
# Step 1: Setup LLM (Groq)
# =========================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Check your .env file.")


def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # ✅ updated model
        temperature=0.5
    )


# =========================
# Step 2: Prompt
# =========================

CUSTOM_PROMPT_TEMPLATE = """
Use the context below to answer the question.
If you don't know the answer, say you don't know.
Do not make up answers.

Context:
{context}

Question:
{input}

Answer:
"""


def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "input"]
    )


# =========================
# Step 3: Load FAISS DB
# =========================

DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# =========================
# Step 4: Create Chain
# =========================

retriever = db.as_retriever(search_kwargs={"k": 5})

llm = load_llm()
prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =========================
# Step 5: Run Query
# =========================

if __name__ == "__main__":
    user_query = input("Write Query Here: ")

    response = qa_chain.invoke(user_query)

    print("\nRESULT:\n", response)