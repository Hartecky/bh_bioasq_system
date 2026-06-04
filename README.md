# BioASQ Agentic QA System

A local, fully offline Question Answering system built on top of the [BioASQ benchmark](https://www.nature.com/articles/s41597-023-02068-4) — a manually curated corpus for biomedical question answering.

The system uses a **RAG (Retrieval-Augmented Generation)** architecture with an agentic routing layer, running entirely on local hardware with no external API dependencies.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

---

## Overview

BioASQ Task B consists of biomedical questions formulated by domain experts, paired with relevant PubMed snippets and reference answers. Questions come in four types:

| Type | Description | Example |
|------|-------------|---------|
| `yesno` | Binary yes/no answer | "Is BRCA1 a tumor suppressor gene?" |
| `factoid` | Short factual answer | "What is the mechanism of action of imatinib?" |
| `list` | List of entities | "List the side effects of metformin." |
| `summary` | Narrative summary | "Describe the role of p53 in apoptosis." |

The system answers each question type using a dedicated prompt strategy, retrieving relevant biomedical context from a FAISS vector index built from BioASQ snippets.

---

## Architecture

```
User Question (free text)
        │
        ▼
┌─────────────────┐
│ QuestionRouter  │  Classifies question type using regex patterns
│                 │  → "yesno" | "factoid" | "list" | "summary"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retriever    │  Embeds question with SentenceTransformer
│                 │  Searches FAISS index for top-k relevant snippets
│  all-MiniLM-L6  │  (built from BioASQ training snippets)
│  + FAISS index  │
└────────┬────────┘
         │  top-k snippets
         ▼
┌─────────────────┐
│  PromptBuilder  │  Builds type-specific prompt:
│                 │  context + question + instructions
└────────┬────────┘
         │  formatted prompt
         ▼
┌─────────────────┐
│  OllamaClient   │  Sends prompt to local LLM via Ollama REST API
│                 │  Default: mistral:7b-instruct-q4_0
└────────┬────────┘
         │  generated answer
         ▼
┌─────────────────┐
│    Evaluator    │  Measures quality against golden standard:
│                 │  accuracy (yesno) | lenient match (factoid)
│                 │  F1 score (list)  | ROUGE-2 (summary)
└─────────────────┘
```

### Key Design Decisions

**Why RAG instead of fine-tuning?**
BioASQ provides expert-curated snippets as context for each question. RAG leverages this directly — the LLM answers based on retrieved evidence, not just parametric memory. This improves factual grounding and makes answers traceable to source documents.

**Why local LLM (Ollama)?**
Full offline operation — no API costs, no data privacy concerns, reproducible results. The system is designed to be model-agnostic: swap the model name in `OllamaClient` to benchmark different LLMs.

**Why FAISS with bi-encoder?**
`all-MiniLM-L6-v2` (80MB) provides fast, high-quality semantic embeddings. FAISS `IndexFlatIP` gives exact nearest-neighbor search. The index is built once and saved to disk — subsequent runs load in seconds.

---

## Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Disk | 20 GB free | 50 GB free |
| GPU (VRAM) | None (CPU only) | 4 GB+ (for GPU inference) |

### Software
- Python 3.11–3.13
- [uv](https://github.com/astral-sh/uv) — Python package manager
- [Ollama](https://ollama.com/) — local LLM runtime
- WSL2 (if on Windows)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bh_bioasq_system.git
cd bh_bioasq_system
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install dependencies

```bash
uv sync
```

> **Note for CUDA users (NVIDIA GPU):** PyTorch requires a separate index for CUDA builds. The `pyproject.toml` is already configured with `index-strategy = "unsafe-best-match"` and the PyTorch CUDA index. Run `uv sync` and verify with:
> ```bash
> uv run python -c "import torch; print(torch.cuda.is_available())"
> ```

### 4. Install Ollama and pull a model

```bash
# Install Ollama (Linux / WSL2)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve &

# Pull the default model (~4 GB)
ollama pull mistral:7b-instruct-q4_0

# Optional: pull Bielik (Polish-optimized model, ~6.7 GB)
ollama pull SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M
```

---

## Data Setup

BioASQ data requires free registration at [bioasq.org](http://bioasq.org/).

1. Register and log in at [http://bioasq.org/](http://bioasq.org/)
2. Navigate to **Datasets → Task B**
3. Download the latest training and test sets (e.g. BioASQ-13b)
4. Place files in the project:

```
data/
└── raw/
    ├── BioASQ-training13b/
    │   └── training13b.json
    └── Task13BGoldenEnriched/
        ├── 13B1_golden.json
        └── ...
```

### Build the FAISS index

Run once — takes ~60 seconds for the full training set:

```bash
uv run python -c "
from src.data.dataset import BioASQDataset
from src.retrieval.retriever import Retriever

ds = BioASQDataset('data/raw/BioASQ-training13b/training13b.json')
retriever = Retriever()
retriever.build(ds.questions)
retriever.save('data/index')
print('Index saved.')
"
```

---

## Usage

### Single question

```python
from src.data.dataset import BioASQDataset
from src.retrieval.retriever import Retriever
from src.generation.llm_client import OllamaClient
from src.agents.pipeline import QAPipeline

# Load index (fast — no rebuild needed)
retriever = Retriever()
retriever.load('data/index')

# Initialize pipeline
llm = OllamaClient(model='mistral:7b-instruct-q4_0')
pipeline = QAPipeline(retriever, llm)

# Answer a question from the dataset
ds = BioASQDataset('data/raw/BioASQ-training13b/training13b.json')
question = ds.filter_by_type('yesno')[0]
answer = pipeline.answer(question)
print(f"Q: {question.body}")
print(f"A: {answer}")
```

### Free-text question (with router)

```python
# For questions without a pre-labeled type — router classifies automatically
answer = pipeline.answer(question, use_router=True)
```

### Switching models

```python
# Use Bielik instead of Mistral
llm = OllamaClient(model='SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M')
pipeline = QAPipeline(retriever, llm)
```

---

## Evaluation

Run evaluation on a sample of questions:

```python
from src.evaluation.evaluator import Evaluator
import json

evaluator = Evaluator(pipeline)
sample = ds.get_sample(50)
metrics = evaluator.evaluate(sample)

print(metrics)

# Save results
with open('experiments/results/baseline_50.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Metrics

| Question Type | Metric | Description |
|---------------|--------|-------------|
| `yesno` | Accuracy | Exact match: "yes" or "no" |
| `factoid` | Lenient Accuracy | Predicted answer contains any gold answer |
| `list` | Mean F1 | Harmonic mean of precision and recall |
| `summary` | ROUGE-2 | Bigram overlap with ideal answer |

### Baseline results (training set, n=50)

> Results will be updated after the baseline evaluation run completes.

---

## Project Structure

```
bh_bioasq_system/
│
├── data/
│   ├── raw/                        # BioASQ JSON files (not tracked by git)
│   ├── processed/                  # Preprocessed data
│   └── index/                      # FAISS index + snippets.pkl
│
├── src/
│   ├── data/
│   │   └── dataset.py              # Question (Pydantic model) + BioASQDataset
│   ├── retrieval/
│   │   └── retriever.py            # Retriever: SentenceTransformer + FAISS
│   ├── generation/
│   │   ├── llm_client.py           # OllamaClient: REST API wrapper
│   │   └── prompt_builder.py       # PromptBuilder: type-specific prompts
│   ├── agents/
│   │   ├── router.py               # QuestionRouter: regex-based classification
│   │   └── pipeline.py             # QAPipeline: main orchestrator
│   └── evaluation/
│       └── evaluator.py            # Evaluator: accuracy, F1, ROUGE-2
│
├── notebooks/                      # Jupyter notebooks for exploration
├── experiments/
│   └── results/                    # JSON files with evaluation results
├── tests/                          # Unit tests
├── pyproject.toml                  # Project dependencies (uv)
└── README.md
```

---

## Roadmap

### Done ✅
- `BioASQDataset` — data loading, filtering, sampling
- `Retriever` — semantic search with SentenceTransformer + FAISS
- `OllamaClient` — local LLM via Ollama REST API
- `PromptBuilder` — type-specific prompt templates
- `QAPipeline` — end-to-end RAG orchestration
- `QuestionRouter` — regex-based question type classification
- `Evaluator` — accuracy, F1, ROUGE-2 metrics

### Next steps 🔜
- [ ] Explore Bielik or other models as an alternative to Ollama
- [ ] Re-ranking with CrossEncoder for better retrieval quality
- [ ] Improve prompts based on baseline evaluation results
- [ ] Fine-tune small model (BioGPT / BioBERT) on BioASQ with QLoRA
- [ ] `ExperimentTracker` — log and compare experiment configurations
- [ ] Unit tests for all components
- [ ] Async batch processing for faster evaluation

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| Data validation | `pydantic` | Type-safe question models |
| Embeddings | `sentence-transformers` | Semantic text representations |
| Vector search | `faiss-cpu` | Fast nearest-neighbor search |
| Local LLM | `ollama` | Run LLMs locally |
| Deep learning | `torch` + `transformers` | Backend for SentenceTransformer (indirect dependency) |
| Evaluation | `rouge-score` | ROUGE metrics for summaries |
| Package manager | `uv` | Fast Python dependency management |

---

## License

MIT

---

*Built as a learning project exploring OOP, RAG architecture, and agentic systems in Python.*