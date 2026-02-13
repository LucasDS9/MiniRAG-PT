import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(path):

    if path.endswith(".pdf"):

        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        return text

    elif path.endswith(".txt"):

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError("Formato não suportado. Use PDF ou TXT.")

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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    return chunks


def ingest_document(path):

    print(f"Carregando documento: {path}")

    full_text = load_document(path)

    chunks = adaptive_chunk(full_text)

    print("Número total de chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        print("\n====================")
        print(f"CHUNK {i}")
        print("====================")
        print(chunk)

    return chunks
