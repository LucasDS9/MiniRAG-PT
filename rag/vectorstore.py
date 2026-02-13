import chromadb


def create_collection(name="rag_test", persist_dir="vector_db"):

    client = chromadb.PersistentClient(path=persist_dir)

    collection = client.get_or_create_collection(name=name)

    print("Collection carregada/criada.")

    return collection


def add_to_collection(collection, chunks, embeddings):

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"id_{i}" for i in range(len(chunks))]
    )

    print("Documentos adicionados ao Chroma.")


def query_collection(collection, query_embedding, n_results=3):

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]

    return retrieved_chunks
