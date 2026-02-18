# 📄 Mini RAG PRO – Sistema de Perguntas e Respostas com RAG

## 🏹 Visão Geral

O Mini RAG PRO é um sistema completo de Retrieval-Augmented Generation (RAG) desenvolvido em Python que permite realizar perguntas sobre documentos PDF utilizando embeddings semânticos, banco vetorial persistente, re-ranking neural e Large Language Models para geração de respostas contextualizadas em português.

O projeto implementa uma arquitetura moderna de IA generativa semelhante às utilizadas em aplicações reais como assistentes corporativos, sistemas jurídicos, buscadores semânticos e chatbots especializados em documentos. Ele cobre todo o ciclo de vida de um sistema RAG: ingestão de documentos, chunking adaptativo, geração de embeddings, indexação vetorial, recuperação semântica, re-ranking, geração de resposta e avaliação quantitativa.

A arquitetura integra ferramentas essenciais do ecossistema moderno de NLP. O ChromaDB é utilizado como banco vetorial persistente para indexação e busca por similaridade semântica. O LangChain Text Splitters é utilizado para realizar chunking adaptativo de documentos, garantindo melhor granularidade na recuperação. A biblioteca Transformers é responsável pelo carregamento e execução dos modelos generativos.

Os seguintes modelos de linguagem foram utilizados no projeto:

- **all-MiniLM-L6-v2**: utilizado para gerar embeddings densos tanto dos chunks dos documentos quanto das perguntas do usuário. Esses vetores semânticos permitem calcular similaridade e recuperar os trechos mais relevantes no ChromaDB, compondo a etapa de *Retrieval* do sistema.

- **Qwen/Qwen2.5-0.5B-Instruct**: utilizado como modelo gerador (LLM) responsável por produzir a resposta final em linguagem natural. Ele recebe o contexto recuperado do banco vetorial junto com a pergunta do usuário e gera uma resposta coerente, objetiva e em português. É o componente responsável pela etapa de *Generation* na arquitetura RAG.
Usado para calcular métricas padrões de llms.
**Resultados :**
Mean F1@3: 0.5111

Mean MRR: 1.0000

Mean nDCG@3: 0.6941

Latência média: 4.329614

**queries** : "Explique como a IA usa matemática para processar informação",
    "Como redes neurais artificiais se inspiram no cérebro humano?",
    "Quais tarefas as máquinas realizam melhor que os humanos?"

{'relevancia': {'relevancia': 8}, 'fidelidade': {'fidelidade': 8}, 'correcao': {'correcao': 9}, 'completude': {'completude': 8.5}}
Para a query **"Explique redes neurais"**


- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** : Utilizado para verificar métricas BLEU e ROUGE.

BLEU médio: 0.2255

ROUGE-1 médio: 0.5746

ROUGE-2 médio: 0.4078

ROUGE-L médio: 0.527

para as queries : "Quais são os pontos em comum entre humanos e máquinas?" e "A maquina tem metacognição?"



---

## 🎯 Importância do Projeto

Este projeto demonstra domínio prático de arquitetura RAG moderna, combinando conceitos de NLP, embeddings densos, vector databases, re-ranking neural e Large Language Models. Diferente de implementações simplificadas, ele inclui um módulo estruturado de avaliação com métricas clássicas de Information Retrieval como Precision@K, Recall@K, F1@K, NDCG@K, MMR e também avaliação com LLM como juiz.

A presença desse módulo de avaliação é um diferencial relevante para aplicações profissionais, pois evidencia preocupação com qualidade de recuperação, mensuração objetiva de performance e análise crítica do sistema. Isso aproxima o projeto de ambientes reais de produção e pesquisa aplicada em IA.

---

## ⚙️ Requisitos e Como Rodar

Recomenda-se Python 3.10 ou superior.

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Principais bibliotecas necessárias:

- sentence-transformers  
- transformers  
- torch  
- chromadb  
- streamlit  
- PyPDF2  
- langchain-text-splitters  
- scikit-learn  
- numpy  
- pandas  

Para melhor desempenho, recomenda-se utilização de GPU ao carregar modelos maiores.

### Executar localmente

```bash
streamlit run app.py
```

Após executar, a aplicação abrirá automaticamente no navegador. O fluxo de uso consiste em enviar PDFs, processar os documentos e realizar perguntas sobre o conteúdo indexado.

---
## 📁 Estrutura do Projeto
```
├── data
│   ├── uploads/ # PDFs enviados pelo usuário
│   ├── vector_db/ # Banco vetorial persistente (ChromaDB)
│
├── rag
│   ├── embedding.py # Carregamento e geração de embeddings
│   ├── ingestion.py # Ingestão e chunking adaptativo
│   ├── llm.py # Carregamento e geração de respostas com LLM
│   └── pipeline.py # Pipeline Mini RAG modular
│   └── vectorstore.py # Integração com ChromaDB
│
├
app.py # Interface Streamlit
evaluation.ipynb # Avaliação do sistema (métricas IR + LLM as Judge)
Pensamento_maquina.pdf # Documento exemplo para testes locais
README.md #descrição do projeto
requirements.txt # Dependências do projeto


```
---

## 🗃️ Descrição dos Módulos

### embedding.py

Responsável por carregar o modelo de embeddings utilizando SentenceTransformers e gerar vetores densos a partir dos chunks de texto. Utiliza por padrão o modelo `all-MiniLM-L6-v2`, equilibrando desempenho e eficiência computacional.

### ingestion.py

Gerencia o carregamento de documentos PDF ou TXT e implementa chunking adaptativo com RecursiveCharacterTextSplitter. O tamanho dos chunks é definido dinamicamente com base na estrutura média do texto, buscando melhorar a qualidade da recuperação semântica.

### llm.py

Carrega o modelo generativo (Qwen Instruct por padrão) e implementa a geração de respostas condicionadas ao contexto recuperado. Utiliza template de chat e instrução explícita para responder sempre em português.

### vectorstore.py

Realiza a integração com o ChromaDB como banco vetorial persistente. Permite criar collections, adicionar documentos com embeddings e realizar consultas por similaridade vetorial.

### pipeline.py

Define o fluxo modular do Mini RAG. Gera embedding da pergunta, consulta o banco vetorial, constrói o contexto a partir dos chunks recuperados e chama o LLM para gerar a resposta final.

### app.py

Interface interativa construída com Streamlit. Permite upload de múltiplos PDFs, indexação automática, re-ranking com CrossEncoder, geração de respostas com efeito de streaming e persistência dos dados no ChromaDB.

### evaluation.ipynb

Módulo de avaliação do sistema. Implementa métricas de recuperação como Cosine Similarity, Precision@K, Recall@K, F1@K, NDCG@K e MMR, além de avaliação utilizando LLM como juiz para análise qualitativa das respostas geradas. Representa um diferencial técnico relevante do projeto.

---

## 🛠️ Ferramentas Utilizadas

| Ferramenta | Finalidade |
|------------|------------|
| Python | Linguagem principal |
| SentenceTransformers | Geração de embeddings densos |
| Transformers (HuggingFace) | Carregamento do LLM |
| Qwen Instruct | Modelo gerador de respostas |
| ChromaDB | Banco vetorial persistente |
| CrossEncoder | Re-ranking neural |
| LangChain Text Splitters | Chunking adaptativo |
| PyPDF2 | Extração de texto de PDFs |
| Streamlit | Interface web |
| HuggingFace Spaces | Deploy da aplicação |
| Scikit-learn | Métricas e similaridade cosseno |

---

## Conclusão

O Mini RAG PRO é um projeto completo que implementa uma arquitetura moderna de Retrieval-Augmented Generation com pipeline estruturado, banco vetorial persistente, re-ranking neural e módulo robusto de avaliação quantitativa e qualitativa.

O sistema demonstra conhecimento sólido em NLP, embeddings densos, recuperação semântica, métricas de Information Retrieval e integração prática com Large Language Models, aproximando-se de padrões utilizados em aplicações reais de IA generativa.

