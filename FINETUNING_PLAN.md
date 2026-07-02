# Qwen Fine-Tuning Implementation Plan

## Project Overview

This document outlines the strategy for fine-tuning a local Qwen model (currently Qwen2.5:14b in Ollama) using Knowledge Distillation from a powerful LLM to improve domain-specific performance on EV Intelligence tasks.

**Starting Point:** 42 human-validated Q&A pairs  
**Goal:** Create a dataset of 500-1000+ synthetic Q&A pairs, validate them, and fine-tune the local model  
**Target Model:** Qwen2.5:14b (via Ollama)  
**Fine-tuning Method:** QLoRA (Parameter-Efficient Fine-Tuning)

---

## Current System State

### What We Have
- **Knowledge Base:** `kb/GNEM_Excel_Data.xlsx` (~76KB)
- **Initial Dataset:** `kb/Human validated questions.xlsx` (42 Q&A pairs)
- **Runtime:** Ollama with Qwen2.5:14b model
- **Storage:** PostgreSQL + pgvector for retrieval
- **Retrieval:** Hybrid (BM25 + dense vector) with cross-encoder reranking

### Current Limitations
- Only 42 examples → insufficient for meaningful fine-tuning
- Local Qwen2.5:14b may lack domain-specific EV knowledge
- No fine-tuning pipeline in place

---

## Implementation Phases

### Phase 1: Data Augmentation (Generate Synthetic Q&A Pairs)

**Objective:** Expand 42 validated questions into 500-1000 synthetic examples.

#### 1.1 Question Paraphrasing
- **Input:** 42 validated questions
- **Process:**
  - Use a powerful external LLM (GPT-4o, Gemini 1.5 Pro) via API
  - For each question, generate 5-10 variations across:
    - Formal to informal tones
    - Short keyword searches vs. conversational queries
    - Different phrasings with same intent
- **Output:** ~350-500 paraphrased Q&A pairs
- **File:** `data_augmentation.py`

#### 1.2 KB-Driven Question Generation
- **Input:** Raw KB text chunks + validated questions
- **Process:**
  - Chunk the Knowledge Base into meaningful segments
  - Use external LLM to generate 3-5 diverse questions per KB segment
  - Ask LLM to provide question + detailed answer grounded in KB text
  - Also generate adversarial "I don't know" questions
- **Output:** ~500-1000 KB-driven Q&A pairs
- **File:** `kb_question_generator.py`

**Key Prompts:**
```
[For paraphrasing]
"Here is a verified question: '{question}'. 
Generate 5 variations of this question ranging from formal to informal, 
short keyword searches, and conversational tones. 
Keep the core intent identical."

[For KB-driven generation]
"Read the following text block from our Knowledge Base:
{kb_text}

Generate 3 diverse, complex questions that can be answered using ONLY this text. 
Provide the question and the detailed answer grounded in the text."
```

---

### Phase 2: Validation & Filtering (LLM-as-Judge)

**Objective:** Ensure synthetic data is accurate and grounded in the KB.

#### 2.1 Accuracy Scoring
- **Input:** Generated (Question, Answer, KB_Source) triplets
- **Process:**
  - Use external LLM to evaluate each pair:
    - Accuracy (1-5): Does the answer contradict the KB?
    - Completeness (1-5): Does it fully answer the question?
    - Relevance (1-5): Is the question domain-specific?
  - Discard pairs scoring < 4 on any dimension
- **Output:** Filtered, high-quality dataset
- **File:** `validation.py`

#### 2.2 Coverage Analysis
- **Input:** Generated questions + KB vocabulary
- **Process:**
  - Extract topics directly from generated questions and raw KB fields
  - Map topics against KB sections
  - Identify gaps in coverage
  - Flag topic clusters with low question density
- **Output:** Coverage report + identified gaps
- **File:** `coverage_analyzer.py`

#### 2.3 Deduplication
- **Input:** All generated pairs
- **Process:**
  - Semantic deduplication using embeddings
  - Remove near-duplicate questions (cosine similarity > 0.85)
- **Output:** Final, deduplicated dataset
- **File:** `deduplication.py`

---

### Phase 3: Dataset Preparation

**Objective:** Format data for fine-tuning in Qwen-compatible ChatML format.

#### 3.1 Format Conversion
- **Input:** Validated Q&A pairs (from Excel, JSON, or DataFrame)
- **Output:** JSONL with ChatML format
- **File:** `dataset_formatter.py`

**Output Format:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant specialized in Georgia EV Intelligence, providing accurate answers about electric vehicles, charging infrastructure, and related topics."
    },
    {
      "role": "user",
      "content": "What is the charging time for a Tesla Model 3?"
    },
    {
      "role": "assistant",
      "content": "The charging time for a Tesla Model 3 depends on the charger type..."
    }
  ]
}
```

#### 3.2 Train/Validation Split
- 80% training, 20% validation
- Stratified split by topic/keyword
- File: `dataset_formatter.py`

---

### Phase 4: Fine-Tuning (QLoRA with Unsloth)

**Objective:** Fine-tune Qwen2.5:14b using Parameter-Efficient Fine-Tuning.

#### 4.1 Training Infrastructure
- **Framework:** Unsloth (optimized for Qwen/Llama)
- **Method:** QLoRA with LoRA adapters
- **File:** `qwen_finetuner.py`

**Key Hyperparameters:**
```python
{
    "epochs": 2,
    "learning_rate": 2e-4,
    "lora_rank": 32,
    "lora_alpha": 64,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_length": 2048,
    "save_steps": 500
}
```

#### 4.2 Adapter Merging
- After training, merge LoRA adapters into base model
- Save fine-tuned model checkpoint
- Export for Ollama integration (if needed)

#### 4.3 Evaluation
- Run validation set through fine-tuned model
- Compare answers with golden answers
- Compute BLEU, ROUGE, semantic similarity metrics
- File: `evaluation.py`

---

### Phase 5: Integration & Deployment

**Objective:** Deploy fine-tuned model and measure improvement.

#### 5.1 Ollama Integration
- Export fine-tuned model to HF format
- Create Modelfile for Ollama
- Load and test in existing RAG pipeline

#### 5.2 Comparison
- Run A/B tests: base Qwen2.5 vs. fine-tuned
- Use existing hybrid retrieval + reranking
- Measure accuracy on held-out test set

#### 5.3 Monitoring
- Log metrics: latency, accuracy, token usage
- Create dashboard (optional)

---

## Directory Structure

```
Finetuning/
├── __init__.py
├── __main__.py
├── data_augmentation.py                 # Paraphrasing + raw-KB generation
├── validation.py                        # LLM-as-judge scoring
├── dataset_formatter.py                 # ChatML conversion
├── llm_client.py                        # LLM provider clients
├── config.py                            # Configuration and paths
├── cli.py                               # Command-line interface
├── kb/
│   ├── GNEM_Excel_Data.xlsx
│   └── Human validated questions.xlsx
├── outputs/
│   └── finetuning/                      # Output datasets & models
│       ├── augmented_questions.jsonl
│       ├── validated_questions.jsonl
│       ├── train_dataset.jsonl
│       ├── val_dataset.jsonl
│       └── checkpoints/
│           └── qwen_finetuned/
└── requirements.txt
```

---

## Key Dependencies

Add to `requirements.txt`:
```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
transformers>=4.38.0
peft>=0.7.0
bitsandbytes>=0.43.0
torch>=2.0.0
xformers>=0.0.25
trl>=0.7.0
```

For external LLM calls (data generation):
```
openai>=1.0.0            # For GPT-4o
google-generativeai>=0.3.0  # For Gemini
```

---

## Timeline & Milestones

| Phase | Tasks | Duration | Owner |
|-------|-------|----------|-------|
| 1 | Data Augmentation | 2-3 days | Claude |
| 2 | Validation & Filtering | 1-2 days | Claude |
| 3 | Dataset Preparation | 1 day | Claude |
| 4 | Fine-tuning | 2-3 days | Claude |
| 5 | Integration & Testing | 1-2 days | Claude |
| **Total** | | **~1-2 weeks** | |

---

## Success Metrics

1. **Dataset Size:** ≥ 500 synthetic Q&A pairs generated
2. **Quality:** ≥ 80% of generated pairs score 4-5 on accuracy
3. **Coverage:** All major KB topics represented in dataset
4. **Fine-tuning:** Model trains without OOM on 16GB GPU
5. **Accuracy:** Fine-tuned model outperforms base model on test set
   - Target: +5-10% improvement in answer accuracy

---

## Rollback & Risk Mitigation

- **Backup:** Keep base model checkpoint before fine-tuning
- **Gradual Rollout:** A/B test fine-tuned model on subset of queries first
- **Validation:** Always run evaluation before deploying to production
- **Versioning:** Track all model checkpoints and training configs

---

## Notes for Implementation

1. **External LLM API Keys:** Need credentials for GPT-4o or Gemini
   - Store in `.env` file
   - Config: `EXTERNAL_LLM_PROVIDER`, `EXTERNAL_LLM_API_KEY`

2. **Computational Resources:**
   - Training: ~8-16 hours on A100 (or 16GB consumer GPU)
   - Inference: Fine-tuned model can run on original hardware (Ollama compatible)

3. **Data Privacy:**
   - Ensure synthetic data does not leak original KB
   - Use LLM-as-judge to validate grounding

4. **Testing Strategy:**
   - Create `tests/finetuning/` with unit tests for each module
   - Integration tests for end-to-end pipeline
