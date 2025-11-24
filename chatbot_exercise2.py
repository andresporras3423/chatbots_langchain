# chatbot.py
import os, glob
import pandas as pd
import streamlit as st
from typing import Iterable, List
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough         # <-- NUEVO
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# --------------- Config UI ---------------
st.set_page_config(page_title="Chat de Datos de la Empresa", page_icon="📊", layout="wide")
st.title("📊 Chat de Datos (Excel) — Carpeta específica")

# API Key
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

with st.sidebar:
    st.header("⚙️ Configuración")
    data_dir = st.text_input(
        "Carpeta con Excels de la empresa",
        value="./información local"
    )
    limit_rows = st.number_input("Límite de filas por hoja (0 = sin límite)", min_value=0, value=0, step=100)
    k_results = st.slider("Trozos por pregunta (k)", 1, 2000, 500)
    
    model = st.selectbox("Modelo", ["gpt-4.1"], index=0)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.1)
    show_sources = st.checkbox("Mostrar fuentes usadas", value=True)
    only_context = st.checkbox("Forzar 'solo contexto' (No lo sé si no hay datos)", value=True)

# --- sustituye tu función de iteración por esta ---
def iter_tabular_docs(folder: str, limit: int = 0) -> Iterable[Document]:
    """Lee todos los .xlsx/.xls/.csv del folder y rinde Document por FILA con metadatos."""
    patterns = ("*.xlsx", "*.xls", "*.csv")
    for pat in patterns:
        for path in glob.glob(os.path.join(folder, pat)):
            try:
                if path.lower().endswith((".xlsx", ".xls")):
                    # Excel: todas las hojas
                    sheets = pd.read_excel(path, sheet_name=None)
                    for sheet_name, df in sheets.items():
                        if df is None or df.empty:
                            continue
                        df = df.fillna("")
                        df.columns = [str(c).strip() for c in df.columns]
                        if limit and limit > 0:
                            df = df.head(limit)
                        for i, row in df.iterrows():
                            parts = []
                            for col, val in row.items():
                                s = str(val)
                                if len(s) > 200:
                                    s = s[:200] + "…"
                                parts.append(f"{col}: {s}")
                            text = (
                                f"FILE: {os.path.basename(path)} | SHEET: {sheet_name} | ROW: {int(i)}\n"
                                + " | ".join(parts)
                            )
                            yield Document(
                                page_content=text,
                                metadata={
                                    "source": os.path.basename(path),
                                    "sheet": sheet_name,
                                    "row": int(i),
                                },
                            )
                else:
                    # CSV: detecta separador y codificación común
                    read_kwargs = dict(sep=None, engine="python", dtype=str)
                    if limit and limit > 0:
                        read_kwargs["nrows"] = limit
                    try:
                        df = pd.read_csv(path, encoding="utf-8-sig", **read_kwargs)
                    except Exception:
                        df = pd.read_csv(path, encoding="latin-1", **read_kwargs)

                    if df is None or df.empty:
                        continue
                    df = df.fillna("")
                    df.columns = [str(c).strip() for c in df.columns]

                    for i, row in df.iterrows():
                        parts = []
                        for col, val in row.items():
                            s = str(val)
                            if len(s) > 200:
                                s = s[:200] + "…"
                            parts.append(f"{col}: {s}")
                        text = (
                            f"FILE: {os.path.basename(path)} | SHEET: CSV | ROW: {int(i)}\n"
                            + " | ".join(parts)
                        )
                        yield Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(path),
                                "sheet": "CSV",
                                "row": int(i),
                            },
                        )
            except Exception as e:
                st.warning(f"❗ No se pudo leer {os.path.basename(path)}: {e}")
                continue


def build_index(docs: Iterable[Document]) -> tuple[InMemoryVectorStore, int]:
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = InMemoryVectorStore(emb)
    batch, total = [], 0
    for d in docs:
        batch.append(d); total += 1
        if len(batch) >= 1000:
            vs.add_documents(batch); batch = []
    if batch: vs.add_documents(batch)
    return vs, total

# Botones
col_a, col_b = st.columns([1,1])
with col_a:
    do_index = st.button("📥 Indexar carpeta ahora")
with col_b:
    clear_index = st.button("🧹 Borrar índice / reiniciar")

if clear_index:
    for key in ["vectorstore", "retriever", "stats", "chat_history"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Índice y memoria limpiados."); st.rerun()

if do_index:
    if not os.path.isdir(data_dir):
        st.error("La ruta indicada no es una carpeta válida.")
    else:
        with st.spinner("Leyendo archivos y creando índice..."):
            docs_iter = iter_tabular_docs(data_dir, limit_rows)
            vs, total_rows = build_index(docs_iter)   # your function returns (vectorstore, row_count)
            st.session_state["vectorstore"] = vs

            # count xlsx/xls/csv files in the folder
            files_count = sum(
                len(glob.glob(os.path.join(data_dir, pat)))
                for pat in ("*.xlsx", "*.xls", "*.csv")
            )
            st.session_state["stats"] = {"files": files_count, "rows": total_rows}

        st.success("✅ Carpeta indexada.")
        st.caption(f"Indexados: {files_count} archivos, {total_rows} filas")


# --------------- Cadena de Chat con RAG ---------------
if "vectorstore" not in st.session_state:
    st.info("💡 Indica la carpeta en la barra lateral y pulsa **Indexar** para empezar.")
    st.stop()

vs: InMemoryVectorStore = st.session_state["vectorstore"]
retriever = vs.as_retriever(search_kwargs={"k": k_results}, search_type="mmr")


system_rules = (
    "Actúas como analista experto en los datos de la empresa. "
    "Responde SIEMPRE en español. "
)
if only_context:
    system_rules += (
        "Realiza cálculos cuando te los pidan, siempre que sea posible con la información proporcionada. "
        "Responde solo preguntas relacionadas con la empresa. "
    )
system_rules += (
    "Cuando la pregunta sea tabular (productos/facturas/clientes), sintetiza en tablas Markdown. "
    "Incluye campos clave (ej. id, nombre, fecha, monto, cliente) cuando existan."
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_rules + "\n\nContexto:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model=model, temperature=temperature)
base_chain = prompt | llm | StrOutputParser()
history = StreamlitChatMessageHistory(key="chat_history")

def format_docs(docs: List[Document]) -> str:
    lines, seen = [], set()
    for d in docs:
        tag = (d.metadata.get("source"), d.metadata.get("sheet"))
        head = f"[{d.metadata.get('source')}]({d.metadata.get('sheet')}) ROW={d.metadata.get('row')}"
        body = d.page_content
        if tag not in seen:
            seen.add(tag)
        if len(body) > 1200:
            body = body[:1200] + "…"
        lines.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(lines)

# ⬇️ Mantiene todas las claves (input, chat_history, etc.) y añade 'context'
chain_core = RunnablePassthrough.assign(
    context=itemgetter("input") | retriever | format_docs
) | base_chain

chain = RunnableWithMessageHistory(
    chain_core,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# Render historial
for m in history.messages:
    with st.chat_message("user" if m.type == "human" else "assistant"):
        st.write(m.content)

# Input usuario
q = st.chat_input("Escribe tu pregunta (ej. 'Total facturado por cliente X en 2024')")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        resp = chain.invoke(
            {"input": q},   # <-- dict con la clave 'input'
            config={"configurable": {"session_id": "empresa"}}
        )
        st.write(resp)

        if show_sources:
            docs = retriever.invoke(q)
            with st.expander("🔎 Fuentes recuperadas"):
                for i, d in enumerate(docs, 1):
                    st.write(f"**{i}.** {d.metadata.get('source')} | {d.metadata.get('sheet')} | fila {d.metadata.get('row')}")
                    st.code(d.page_content[:800] + ("…" if len(d.page_content) > 800 else ""))
