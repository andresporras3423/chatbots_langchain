import os
import streamlit as st
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ---------------------- Configuración básica UI ----------------------
st.set_page_config(page_title="Demo RAG simple", page_icon="💬", layout="wide")
st.title("💬 Demo RAG con LangChain (memoria en código, sin historial)")

# API Key
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

if not os.getenv("OPENAI_API_KEY"):
    st.info("🔑 Ingresa tu API key para continuar.")
    st.stop()

# ---------------------- Sidebar: configuración del modelo ----------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    model = st.selectbox("Modelo", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.1)
    k_results = st.slider("Trozos por pregunta (k)", 1, 5, 2)


# ---------------------- 1) Base de conocimiento (TU LISTA) ----------------------
documents = [
    "LangChain allows the development of conversational applications with LLMs.",
    "LLMs are language models trained with large volumes of text.",
    "ChromaDB is a vector database that allows storing embeddings."
]

# ---------------------- 2) Embeddings + VectorStore (en memoria) ----------------------
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vs = InMemoryVectorStore(emb)
vs.add_texts(documents)
retriever = vs.as_retriever(search_kwargs={"k": k_results})

# ---------------------- 3) LLM ----------------------
llm = ChatOpenAI(model=model, temperature=temperature)

# ---------------------- 4) Prompt y cadena RAG (sin memoria) ----------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Responde en español usando SOLO el contexto. "
        "Si no hay datos suficientes, di: 'No lo sé'.\n\n"
        "Contexto:\n{context}"
    ),
    ("user", "{question}")
])


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# Esto es literalmente:
# {"context": retriever|format_docs, "question": Passthrough} | prompt | llm | StrOutputParser()
chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------- 5) Input del usuario ----------------------
q = st.chat_input("Haz una pregunta sobre la base de conocimiento (ej. '¿Qué es ChromaDB?')")
if q:
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        resp = chain.invoke({"question": q})
        st.write(resp)
