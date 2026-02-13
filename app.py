import os
import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "Qwen/Qwen2-0.5B-Instruct"

UPLOAD_DIR = "data/uploads"
VECTOR_DB_DIR = "data/vector_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_reranker():
    return CrossEncoder(RERANK_MODEL)

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    return tokenizer, model


@st.cache_resource
def load_vectorstore():
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    return client.get_or_create_collection(name="rag_collection")


def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def adaptive_chunk(text):

    avg_len = len(text.split()) / max(text.count("."), 1)

    if avg_len > 20:
        chunk_size = 800
        overlap = 150
    elif avg_len > 10:
        chunk_size = 500
        overlap = 100
    else:
        chunk_size = 300
        overlap = 50

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_text(text)


def ingest_documents(files, embedder, collection):

    for file in files:

        save_path = os.path.join(UPLOAD_DIR, file.name)

        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        text = load_pdf_text(save_path)
        chunks = adaptive_chunk(text)

        embeddings = embedder.encode(chunks).tolist()

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{file.name}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": file.name}] * len(chunks)
        )


def rerank_chunks(query, chunks, reranker, top_k=3):

    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk for chunk, _ in ranked[:top_k]]


def generate_answer_stream(prompt, tokenizer, model):

    messages = [
        {"role": "system", "content": "Responda sempre em português e seja objetivo."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    placeholder = st.empty()

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    partial = ""
    for word in response.split():
        partial += word + " "
        placeholder.markdown(partial)

    return response


def rag_query(query, embedder, collection, tokenizer, model, reranker):

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    retrieved_chunks = results["documents"][0]

    best_chunks = rerank_chunks(
        query,
        retrieved_chunks,
        reranker,
        top_k=3
    )

    context = "\n".join(best_chunks)

    prompt = f"""
Use apenas o contexto para responder.

Contexto:
{context}

Pergunta:
{query}
"""

    return generate_answer_stream(prompt, tokenizer, model)


st.title("📄 Mini RAG PRO - Chat com seus PDFs")

embedder = load_embedding_model()
reranker = load_reranker()
tokenizer, model = load_llm()
collection = load_vectorstore()

uploaded_files = st.file_uploader(
    "Envie PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write("Arquivos enviados:")
    for f in uploaded_files:
        st.write(f"📄 {f.name} ({round(f.size/1024,2)} KB)")

if uploaded_files and st.button("Processar documentos"):
    with st.spinner("Processando documentos..."):
        ingest_documents(uploaded_files, embedder, collection)
    st.success("Documentos adicionados!")

query = st.text_input("Faça uma pergunta sobre os documentos")

if query:
    with st.spinner("Pensando..."):
        rag_query(
            query,
            embedder,
            collection,
            tokenizer,
            model,
            reranker
        )
