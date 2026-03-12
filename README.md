## WWII RAG Assistant ##

Interactive AI assistant that answers questions about World War II using a Retrieval-Augmented Generation (RAG) pipeline built from documentary video transcripts.

The system retrieves relevant transcript fragments from YouTube videos and generates grounded answers with citations and timestamps pointing to the original sources.

---

# Demo

The assistant supports:

- Text questions
- Voice questions (speech-to-text)
- Optional voice responses (text-to-speech)
- Source citations with timestamps linking to the original video fragment

Example question:

> What happened during the D-Day?

The system retrieves transcript fragments from historical documentaries and generates a grounded answer based on those sources.

---

# System Architecture

The project implements a hybrid Retrieval-Augmented Generation pipeline.
    1-User Question
    2-Query Rewriting
    3-Hybrid Retrieval (FAISS + BM25)
    4-Cross-Encoder Reranking
    5-Context Construction
    6-LLM Answer Generation
    7-Sources with Video Timestamps

---

# Data Pipeline

The knowledge base is built from documentary video transcripts.
    1-YouTube Videos
    2-Transcript Ingestion
    3-Transcript Chunking
    4-Metadata Enrichment
    5-Canonical Chunk Dataset
    6-Vector Databases


Final dataset: 
    data/chunks/all_chunks_stable.json


Each chunk contains:

- transcript text
- timestamps
- video metadata
- canonical URLs to the original video

---

# Retrieval Architecture

The assistant uses hybrid retrieval (FAISS + BM25).

| System | Role |
|------|------|
| FAISS | semantic similarity search |
| BM25 | lexical keyword search |
| Chroma | persistent vector database |

Retrieved passages are then reranked using a cross-encoder model to improve precision.

---

# Evaluation

The project includes an automated evalaution workflow with LangSmith.

Evaluation pipeline:
    1-Canonical chunk dataset
    2-Automatic question generation
    3-LangSmith dataset creation
    4-RAG evaluation across backends
    5-Error analysis and experiment comparison

The evaluation scripts allow:

    -automatic generation of candidate QA examples from transcript chunks
    -upload of evaluation datasets to LangSmith
    -comparison of multiple retrieval backends (faiss, chroma, hybrid)
    -automatic scoring with LLM-as-judge evaluators
    -post-hoc error analysis and report generation

Main evaluation scripts:

src/evals/
    01_generate_eval_questions.py
    02_build_langsmith_dataset.py
    03_run_langsmith_eval.py
    04_make_error_analysis_table.py

Evaluated metrics:

    -Correctness: semantic agreement with the reference answer.
    -Groundedness: support of the generated answer in the retrieved context.

## RUNNING THE PROJECT

# 1. Install dependencies

bash
    pip install -r requirements.txt

# 2. Configure enviroment variables:

Create a .env file in the project root:

    OPENAI_API_KEY=your_api_key
    EMBED_MODEL=text-embedding-3-large

# 3. Buiding the knowledge base:

Run the data pipeline:

Bash 
    python src/01_gather_all.py
    python src/02_build_canonical_chunks.py
    python src/03_build_vectorstore.py
    python src/03b_build_chroma_store.py 

# 4. Launch the assitant:

Ran the Gradio interface:

Bash
    python run_app.py

The interface will open in your browser.

---

## RUNNING THE EVALUATION

1-Generate evaluation candidates:
    
    Bash:
        python -m src.evals.01_generate_eval_questions
            #creates: data/evals/langsmith_eval_candidates.json

2-Upload examples to LangSmith:

    Bash:
        python -m src.evals.02_build_langsmith_dataset

3-Run automated evalaution:

    Bash:
        python -m src.evals.run_langsmith_eval
            #the evauation can compare (by default):
                -Faiss
                -Chroma
                -Hybrid

 4-Generate diagnostic reports:

    Bash:
        python -m src.evals.04_make_error_analysis_table            


## TECHNOLOGIES USED

Main components:
    -Python
    -LangChain
    -OpenAI API
    -FAISS
    -ChromaDB
    -Gradio

Additional tools:
    -OpenAI speech-to-text
    -OpenAI text-to-speech
    -YouTube Transcript API
    -SentenceTransformes (cross-encoder reranking)

## PROJECT STRUCTURE

src/
    agents/
        query_rewriter.py
        retriever_agent.py
        retriever_bm25.py
        reranker_agent.py
        context_builder.py
        answer_agent.py

    pipeline/
        rag_pipeline.py

    app/
        app.py
        voice_utils.py

01_gather_all.py
    01b_gather_all.ipynb(alternative transcript gathering)
02_build_canonical_chunks.py
03_build_vectorstore.py
03b_build_chroma_store.py

data/
    raw/
    chunks/
    chroma/
    vectorstore/

## LIMITATIONS

-Knowledge base limited to selected documentary transcripts.
-Answers depend on transcript coverage.
-Historical nuance may be missing if not present in transcripts.

## DESIGN DECISIONS

donde explicas por qué elegiste:

- RAG
- hybrid retrieval
- reranker
- canonical chunk dataset
- Chroma + FAISS

Eso suele ser lo que más valoran en proyectos de IA.

