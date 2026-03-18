import os
import re
import json
import shutil
import chromadb
import gradio as gr
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   
JUDGE_MODEL     = "Qwen/Qwen2-0.5B-Instruct"              

UPLOAD_DIR    = "data/uploads"
VECTOR_DB_DIR = "data/vector_db"

os.makedirs(UPLOAD_DIR,    exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)


print("Carregando modelos... (pode demorar na primeira vez)")

embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model     = AutoModelForCausalLM.from_pretrained(GEN_MODEL, low_cpu_mem_usage=True)

judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
judge_model     = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, low_cpu_mem_usage=True)

print("Modelos carregados!")


client     = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection(name="rag_collection")


def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text   = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def adaptive_chunk(text: str) -> list[str]:
    word_count = len(text.split())
    sent_count = max(text.count("."), 1)
    avg_len    = word_count / sent_count

    if avg_len > 20:
        chunk_size, overlap = 800, 150
    elif avg_len > 10:
        chunk_size, overlap = 500, 100
    else:
        chunk_size, overlap = 300,  50

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = overlap,
        separators    = ["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


def rerank_chunks(query: str, chunks: list[str], top_k: int = 3) -> list[str]:
    pairs  = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


def run_model(tokenizer, model, messages: list[dict], max_new_tokens: int = 300) -> str:
    """Utilitário genérico para gerar texto com qualquer modelo chat."""
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )
    inputs  = tokenizer(text_input, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature    = 0.7,
        top_p          = 0.9,
        do_sample      = True,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens = True,
    ).strip()


def ingest_documents(files) -> str:
    if not files:
        return "⚠️ Nenhum arquivo enviado."

    inserted_total = 0
    for file in files:
        filename  = os.path.basename(file)
        save_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(file, save_path)

        text = load_pdf_text(save_path)
        if not text.strip():
            continue

        chunks     = adaptive_chunk(text)
        embeddings = embedder.encode(chunks).tolist()
        ids        = [f"{filename}_{i}" for i in range(len(chunks))]

        try:
            collection.delete(ids=ids)
        except Exception:
            pass

        collection.add(
            documents  = chunks,
            embeddings = embeddings,
            ids        = ids,
            metadatas  = [{"source": filename}] * len(chunks),
        )
        inserted_total += len(chunks)

    return f"✅ Documentos processados! {inserted_total} chunks indexados."

_last_context = ""

def rag_query(query: str) -> str:
    global _last_context

    if not query.strip():
        return "⚠️ Digite uma pergunta."

    query_embedding  = embedder.encode([query]).tolist()
    results          = collection.query(query_embeddings=query_embedding, n_results=10)
    retrieved_chunks = results["documents"][0]

    if not retrieved_chunks:
        return "Não encontrei informações relevantes nos documentos enviados."

    best_chunks   = rerank_chunks(query, retrieved_chunks, top_k=3)
    _last_context = "\n\n".join(best_chunks)

    messages = [
        {"role": "system", "content": "Responda sempre em português. Seja objetivo e claro."},
        {"role": "user",   "content": (
            "Use apenas o contexto abaixo para responder. "
            "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
            f"Contexto:\n{_last_context}\n\n"
            f"Pergunta:\n{query}"
        )},
    ]

    return run_model(gen_tokenizer, gen_model, messages, max_new_tokens=300)


def embedding_scores(context: str, response: str) -> dict:
    """
    Fidelidade:  similaridade cosseno entre resposta e contexto.
    Concisão:    penaliza respostas proporcionalmente longas ao contexto.
    """
    r_emb      = embedder.encode([response])
    c_emb      = embedder.encode([context])
    fidelidade = float(cosine_similarity(r_emb, c_emb)[0][0])

    resp_words = len(response.split())
    ctx_words  = len(context.split())
    ratio      = resp_words / max(ctx_words, 1)
    concisao   = float(max(0.0, 1.0 - max(0.0, ratio - 0.4)))

    return {
        "fidelidade_emb": round(fidelidade, 3),
        "concisao_emb":   round(concisao,   3),
    }


def llm_judge(query: str, context: str, response: str) -> dict:
    prompt = f"""Você é um avaliador de respostas de IA. Avalie a resposta abaixo com notas de 1 a 5.

Definições:
- Coerência: a resposta é lógica, clara e bem estruturada?
- Fidelidade: a resposta se baseia apenas no contexto, sem inventar informações?
- Concisão: a resposta é objetiva e sem informações desnecessárias?

Pergunta: {query}
Contexto: {context[:600]}
Resposta: {response}

Retorne SOMENTE JSON válido, sem texto adicional:
{{"coerencia": <1-5>, "fidelidade": <1-5>, "concisao": <1-5>}}"""

    messages = [
        {"role": "system", "content": "Retorne apenas JSON válido."},
        {"role": "user",   "content": prompt},
    ]

    raw = run_model(judge_tokenizer, judge_model, messages, max_new_tokens=60)

    try:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            return {
                "coerencia":  min(5, max(1, int(scores.get("coerencia",  1)))),
                "fidelidade": min(5, max(1, int(scores.get("fidelidade", 1)))),
                "concisao":   min(5, max(1, int(scores.get("concisao",   1)))),
            }
    except Exception:
        pass

    return {"coerencia": 0, "fidelidade": 0, "concisao": 0}


def avaliar_resposta(query: str, response: str) -> str:
    if not query.strip() or not response.strip():
        return "⚠️ Faça uma pergunta e obtenha uma resposta antes de avaliar."
    if not _last_context:
        return "⚠️ Nenhum contexto disponível. Faça uma pergunta primeiro."

    emb = embedding_scores(_last_context, response)
    llm = llm_judge(query, _last_context, response)

    return f"""## Avaliação da Resposta

### LLM-as-Judge (Qwen2 0.5B avalia TinyLlama 1.1B)
| Métrica    | Score (1-5) |
|------------|-------------|
| Coerência  | {llm['coerencia']} |
| Fidelidade | {llm['fidelidade']} |
| Concisão   | {llm['concisao']} |

### Score por Similaridade de Embeddings (0.0-1.0)
| Métrica    | Score |
|------------|-------|
| Fidelidade | {emb['fidelidade_emb']} |
| Concisão   | {emb['concisao_emb']} |

*Gerador: TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Juiz: Qwen/Qwen2-0.5B-Instruct*"""


with gr.Blocks(title="Mini RAG PRO") as demo:
    gr.Markdown("# 📄 Mini RAG PRO — Chat com seus PDFs")
    gr.Markdown("Envie PDFs, processe, pergunte e avalie a qualidade da resposta.")

    with gr.Row():
        with gr.Column(scale=1):
            files         = gr.File(file_types=[".pdf"], file_count="multiple", label="📂 Envie PDFs")
            upload_btn    = gr.Button("⚙️ Processar documentos", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            question = gr.Textbox(
                label       = "💬 Faça uma pergunta",
                placeholder = "Ex: Qual é o tema principal do documento?",
                lines       = 2,
            )
            ask_btn  = gr.Button("🔍 Perguntar", variant="primary")
            answer   = gr.Textbox(label="🤖 Resposta (TinyLlama 1.1B)", lines=6, interactive=False)

            gr.Markdown("---")
            eval_btn    = gr.Button("📊 Avaliar resposta", variant="secondary")
            eval_output = gr.Markdown()

    upload_btn.click(ingest_documents, inputs=files,              outputs=upload_status, show_progress="full")
    ask_btn.click(   rag_query,        inputs=question,           outputs=answer,        show_progress="full")
    question.submit( rag_query,        inputs=question,           outputs=answer,        show_progress="full")
    eval_btn.click(  avaliar_resposta, inputs=[question, answer], outputs=eval_output,   show_progress="full")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)