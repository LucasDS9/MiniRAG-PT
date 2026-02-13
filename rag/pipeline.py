from rag.vectorstore import query_collection
from rag.llm import generate_answer


def mini_rag(query, embedding_model, collection, tokenizer, model, n_results=3):

    print("\nGerando embedding da pergunta...")

    query_embedding = embedding_model.encode([query])

    print("Buscando chunks relevantes...")

    retrieved_chunks = query_collection(
        collection,
        query_embedding,
        n_results=n_results
    )

    context = "\n".join(retrieved_chunks[:2])

    prompt = f"""
Use apenas o contexto para responder.

Contexto:
{context}

Pergunta:
{query}
"""

    print("\nGerando resposta...")

    answer = generate_answer(prompt, tokenizer, model)

    return {
        "query": query,
        "context": context,
        "answer": answer
    }
