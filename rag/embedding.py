from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="all-MiniLM-L6-v2"):

    print("Carregando modelo de embeddings...")
    model = SentenceTransformer(model_name)

    return model


def generate_embeddings(model, chunks):

    print("Gerando embeddings...\n")

    embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=False
)


    return embeddings
