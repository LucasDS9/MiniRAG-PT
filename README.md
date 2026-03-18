# 📄 Mini RAG PRO — Sistema de Perguntas e Respostas com RAG

> Sistema completo de **Retrieval-Augmented Generation (RAG)** desenvolvido em Python que permite realizar perguntas sobre documentos PDF utilizando embeddings semânticos, banco vetorial persistente, re-ranking neural e Large Language Models para geração de respostas contextualizadas em português.

🌐 **[Acesse o portfólio](https://LucasDS9.github.io)** · 🚀 **[Testar o modelo](https://huggingface.co/spaces/LucasDS9/MiniRAG-PT)**

---

## 💼 Aplicações Reais

| Caso de uso | Descrição |
|---|---|
| 🏢 Assistentes corporativos | Q&A sobre documentos internos, manuais e políticas |
| ⚖️ Sistemas jurídicos | Consulta semântica sobre contratos, legislações e pareceres |
| 🔍 Buscadores semânticos | Recuperação inteligente em bases de conhecimento |
| 🤖 Chatbots especializados | Atendimento automatizado com base em documentação própria |

---

## 📌 Sobre o Projeto

O Mini RAG PRO cobre todo o ciclo de vida de um sistema RAG: **ingestão de documentos, estratégias de chunking, geração de embeddings semânticos, indexação vetorial com ChromaDB, recuperação por similaridade, re-ranking neural, geração de resposta com LLM e avaliação quantitativa completa**.

A arquitetura implementada se aproxima dos padrões utilizados em aplicações reais de IA generativa, com atenção especial à qualidade da recuperação e à avaliação objetiva do sistema — diferenciais relevantes em contextos profissionais e de pesquisa aplicada.

**Modelos utilizados:**

- **all-MiniLM-L6-v2** — geração de embeddings densos para chunks e perguntas via SentenceTransformers, compondo a etapa de *Retrieval*
- **Qwen/Qwen2.5-1.5B-Instruct** — modelo gerador responsável pela etapa de *Generation*, produzindo respostas coerentes e objetivas em português com base no contexto recuperado
- **cross-encoder/ms-marco-MiniLM-L-6-v2** — re-ranking neural dos chunks recuperados para maximizar a relevância do contexto enviado ao LLM
- **Qwen/Qwen2.5-0.5B-Instruct** — utilizado no `evaluation.ipynb` como juiz (LLM-as-Judge) pelo baixo custo computacional. Vale destacar que **métricas como relevância, coerência e completude teriam scores mais altos com um modelo juiz mais robusto** — o 0.5B representa um limite inferior conservador de avaliação

---

## 📁 Estrutura do Projeto

```text
📦 MiniRAG-PT
├── 📁 data
│   ├── uploads/          # PDFs enviados pelo usuário
│   └── vector_db/        # Banco vetorial persistente (ChromaDB)
│
├── 📁 rag
│   ├── embedding.py      # Carregamento e geração de embeddings
│   ├── ingestion.py      # Ingestão e chunking adaptativo
│   ├── llm.py            # Carregamento e geração de respostas com LLM
│   ├── pipeline.py       # Pipeline RAG modular
│   └── vectorstore.py    # Integração com ChromaDB
│
├── app.py                # Interface Gradio
├── evaluation.ipynb      # Avaliação completa do sistema RAG
├── requirements.txt
└── README.md
```

---

## 🧱 Etapas do Projeto

### 1️⃣ Ingestão de documentos
- Leitura de documentos PDF com **PyPDF2**
- Extração e limpeza do texto por página

### 2️⃣ Estratégias de chunking
- **Chunking adaptativo** via **LangChain RecursiveCharacterTextSplitter**: tamanho dos chunks e overlap definidos dinamicamente com base na estrutura média do texto, garantindo melhor granularidade na recuperação
- O notebook de avaliação explora e compara **diferentes estratégias** — chunk_size, overlap variável, chunking semântico e chunking por sentença — analisando o impacto de cada abordagem nas métricas de retrieval

### 3️⃣ Geração de embeddings e indexação vetorial
- Transformação dos chunks em **vetores semânticos densos** com **SentenceTransformers (all-MiniLM-L6-v2)**
- Indexação e persistência no **ChromaDB** (banco vetorial local)
- Busca por **cosine similarity** entre o embedding da pergunta e os chunks indexados

### 4️⃣ Recuperação e re-ranking
- Consulta vetorial retorna os `n` chunks mais similares à pergunta do usuário
- **MMR (Maximal Marginal Relevance)** no pipeline de avaliação para diversificar os chunks e reduzir redundância
- **CrossEncoder (ms-marco-MiniLM-L-6-v2)** realiza re-ranking neural para selecionar os chunks mais relevantes antes de enviar ao LLM

### 5️⃣ Geração de resposta
- **Qwen2.5-1.5B-Instruct** recebe o contexto re-rankeado e a pergunta do usuário
- Instrução explícita via prompt para responder sempre em português, com base exclusiva no contexto fornecido
- Suporte a quantização 4-bit em GPU via **BitsAndBytesConfig** para redução de uso de memória

### 6️⃣ Avaliação quantitativa — `evaluation.ipynb`

O notebook de avaliação é um **diferencial técnico relevante** do projeto. Implementa um pipeline completo de avaliação RAG cobrindo desde o chunking até métricas de geração:

**Fluxo:** Documentos → Chunking → Embeddings → ChromaDB → Retrieval → Reranking (MMR) → Prompt → LLM → Avaliação

- **Chunking strategies:** comparação entre diferentes configurações de chunk_size, overlap, chunking semântico e por sentença
- **Métricas de Retrieval (IR):** Recall@K, Precision@K, F1@K, NDCG@K, MRR
- **Similaridade semântica:** cosine similarity entre resposta gerada e contexto recuperado
- **LLM-as-Judge:** avaliação automática de relevância, fidelidade, correção e completude usando **Qwen 0.5B** como juiz
- **Métricas de geração:** BLEU e ROUGE calculados com **Qwen 1.5B (Hugging Face)**
- **Prompt Engineering:** variação e controle de instruções para análise do impacto no output

> ⚠️ Os scores do LLM-as-Judge foram obtidos com **Qwen 0.5B**, modelo de baixo custo computacional. Um modelo juiz mais robusto produziria avaliações mais precisas e scores potencialmente mais altos — os resultados atuais representam um limite inferior conservador.

---

## 📊 Resultados da Avaliação

### Métricas de Retrieval (Information Retrieval)

| Métrica | Valor |
|---|---|
| Mean F1@3 | 0.5111 |
| Mean MRR | 1.0000 |
| Mean nDCG@3 | 0.6941 |
| Latência média | 4.33s |

### LLM-as-Judge — query: *"Explique redes neurais"*

| Dimensão | Score (0–10) |
|---|---|
| Relevância | 8 |
| Fidelidade | 8 |
| Correção | 9 |
| Completude | 8.5 |

### Métricas BLEU e ROUGE

Calculadas com **Qwen 1.5B (Hugging Face)** para as queries:
*"Quais são os pontos em comum entre humanos e máquinas?"* e *"A máquina tem metacognição?"*

| Métrica | Valor |
|---|---|
| BLEU médio | 0.2255 |
| ROUGE-1 médio | 0.5746 |
| ROUGE-2 médio | 0.4078 |
| ROUGE-L médio | 0.5270 |

---

## 🚀 Como Rodar Localmente

Recomenda-se Python 3.10 ou superior.

```bash
pip install -r requirements.txt
python app.py
```

Após executar, a aplicação abrirá no navegador via Gradio. O fluxo consiste em enviar PDFs, processar os documentos e realizar perguntas sobre o conteúdo indexado.

> Para melhor desempenho na geração de respostas, recomenda-se uso de **GPU**.

---

## 🚀 Conclusão

O Mini RAG PRO demonstra conhecimento sólido em NLP, embeddings semânticos, estratégias de chunking, recuperação vetorial, re-ranking neural e integração prática com Large Language Models. O pipeline de avaliação estruturado — cobrindo chunking strategies, métricas de IR, LLM-as-Judge, BLEU e ROUGE — é um diferencial que aproxima o projeto de padrões utilizados em ambientes reais de produção e pesquisa aplicada em IA generativa.

---

## 🛠 Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| 🐍 **Python** | Linguagem principal do projeto |
| 🔤 **SentenceTransformers (all-MiniLM-L6-v2)** | Geração de embeddings semânticos densos |
| 🗄️ **ChromaDB** | Banco vetorial persistente para indexação e busca |
| 🔁 **CrossEncoder (ms-marco-MiniLM-L-6-v2)** | Re-ranking neural dos chunks recuperados |
| 🤖 **Qwen2.5-1.5B-Instruct** | Modelo gerador de respostas (LLM principal) |
| 🧪 **Qwen2.5-0.5B-Instruct** | LLM-as-Judge no pipeline de avaliação |
| 📊 **Qwen 1.5B (Hugging Face)** | Cálculo de métricas BLEU e ROUGE |
| ✂️ **LangChain Text Splitters** | Chunking adaptativo de documentos |
| 📄 **PyPDF2** | Extração de texto de PDFs |
| 📐 **Scikit-learn** | Métricas de IR e similaridade cosseno |
| 🚀 **Gradio** | Interface web da aplicação |
| 🤗 **Hugging Face Spaces** | Deploy e hospedagem |