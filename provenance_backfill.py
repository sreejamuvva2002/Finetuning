"""Reconstruct provenance for the existing augmented dataset (no LLM calls).

The original ``augment_dataset`` wrote records in a fixed order with no provenance:

    [ human originals ] + [ paraphrases ] + [ KB-generated ] + [ adversarial ]

This script recovers that provenance by index and by matching paraphrase answers back
to their trusted human originals, then emits an enriched JSONL where every record carries
``source_type`` / ``source_id`` / ``source_ref`` / ``reference_answer`` / ``trusted`` so the
grounded validator and the group-aware splitter can do their jobs.

It performs NO Ollama calls. Paraphrase intent-verification and KB grounding happen later in
``validation.py``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from . import config
    from .data_augmentation import load_validated_questions
except ImportError:
    import config
    from data_augmentation import load_validated_questions

logger = logging.getLogger(__name__)

ENRICHED_JSONL = config.FINETUNING_OUTPUTS_DIR / "augmented_enriched.jsonl"

# A record counts as an adversarial refusal if its answer looks like a canned "I don't know".
_REFUSAL_RE = re.compile(r"i\s+don'?t\s+have\s+information", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Collapse whitespace so near-identical answers compare equal."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _load_kb_chunks_with_meta() -> list[dict]:
    """Reload KB rows exactly as ``KBQuestionGenerator._load_kb_chunks`` did.

    Returns a list aligned to the KB generation order, each item carrying the chunk text,
    the source row index, and the company name for provenance.
    """
    df = pd.read_excel(config.KB_DATA).dropna(how="all")
    if config.KB_RECORD_LIMIT > 0 and len(df) > config.KB_RECORD_LIMIT:
        df = df.sample(n=config.KB_RECORD_LIMIT, random_state=42)

    chunks = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        fields = []
        for column, value in row.items():
            if pd.notna(value) and str(value).strip():
                fields.append(f"{column}: {str(value).strip()}")
        if not fields:
            continue
        company = row.get("Company")
        chunks.append(
            {
                "chunk": " | ".join(fields),
                "kb_row_index": row_index,
                "kb_company": str(company).strip() if pd.notna(company) else None,
            }
        )
    return chunks


def _load_human_with_num() -> list[dict]:
    """Load human-validated Q&A plus the workbook ``Num`` for stable source_ids."""
    df = pd.read_excel(config.HUMAN_QA_EXCEL)
    columns = {
        str(c).strip().casefold(): c for c in df.columns if pd.notna(c)
    }

    def find(cands):
        for cand in cands:
            if cand.casefold() in columns:
                return columns[cand.casefold()]
        return None

    q_col = find(("Question", "Questions"))
    a_col = find(("Human validated answers", "Human validated answer", "Answer", "Answers"))
    num_col = find(("Num", "No", "Number", "Id"))

    humans = []
    for position, (_, row) in enumerate(df.iterrows()):
        q, a = row.get(q_col), row.get(a_col)
        if pd.isna(q) or pd.isna(a):
            continue
        q, a = str(q).strip(), str(a).strip()
        if not q or not a:
            continue
        num = row.get(num_col) if num_col is not None else None
        num = int(num) if (num is not None and pd.notna(num)) else position + 1
        humans.append({"num": num, "question": q, "answer": a})
    return humans


def backfill(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    """Reconstruct provenance and write the enriched dataset.

    Returns a summary dict with per-segment counts and any dropped records.
    """
    input_path = input_path or config.AUGMENTED_QUESTIONS_JSONL
    output_path = output_path or ENRICHED_JSONL

    records = [json.loads(l) for l in open(input_path) if l.strip()]
    total = len(records)
    logger.info("Loaded %d augmented records from %s", total, input_path)

    humans = _load_human_with_num()
    kb_chunks = _load_kb_chunks_with_meta()
    human_n = len(humans)
    kb_n = len(kb_chunks) * config.KB_QUESTIONS_PER_CHUNK

    # Detect the contiguous adversarial block at the tail.
    adversarial_n = 0
    for rec in reversed(records):
        if _REFUSAL_RE.search(rec.get("answer", "")):
            adversarial_n += 1
        else:
            break

    paraphrase_n = total - human_n - kb_n - adversarial_n
    logger.info(
        "Segment plan: human=%d paraphrase=%d kb=%d adversarial=%d (total=%d)",
        human_n, paraphrase_n, kb_n, adversarial_n, total,
    )
    if paraphrase_n < 0:
        raise ValueError(
            f"Segment math is inconsistent (paraphrase count = {paraphrase_n}). "
            "The augmented file does not match the expected generation order."
        )

    # Boundaries in the flat file.
    h_start, h_end = 0, human_n
    p_start, p_end = h_end, h_end + paraphrase_n
    k_start, k_end = p_end, p_end + kb_n
    a_start, a_end = k_end, total

    # Map normalized human answer -> human meta, for paraphrase linking.
    ans_to_human = {_normalize(h["answer"]): h for h in humans}

    enriched: list[dict] = []
    dropped_paraphrases: list[dict] = []
    seen_para_questions: set[str] = set()
    para_counter: dict[str, int] = {}

    # --- Human originals: trusted, bypass judge ---
    for offset, i in enumerate(range(h_start, h_end)):
        h = humans[offset]
        enriched.append({
            "id": f"human-{h['num']:04d}",
            "question": records[i]["question"],
            "answer": records[i]["answer"],
            "source_type": "human",
            "source_id": f"human-{h['num']:04d}",
            "source_ref": {"human_num": h["num"]},
            "reference_answer": records[i]["answer"],
            "trusted": True,
        })

    # --- Paraphrases: keep only if answer matches a trusted answer; dedup questions ---
    for i in range(p_start, p_end):
        rec = records[i]
        match = ans_to_human.get(_normalize(rec["answer"]))
        norm_q = _normalize(rec["question"])
        if match is None:
            dropped_paraphrases.append({**rec, "_reason": "answer_not_trusted"})
            continue
        if norm_q in seen_para_questions or norm_q == _normalize(match["question"]):
            dropped_paraphrases.append({**rec, "_reason": "duplicate_question"})
            continue
        seen_para_questions.add(norm_q)
        parent_id = f"human-{match['num']:04d}"
        para_counter[parent_id] = para_counter.get(parent_id, 0) + 1
        enriched.append({
            "id": f"{parent_id}-p{para_counter[parent_id]}",
            "question": rec["question"],
            "answer": rec["answer"],
            "source_type": "paraphrase",
            "source_id": f"human-{match['num']:04d}",
            "source_ref": {"human_num": match["num"]},
            "reference_answer": match["answer"],
            "trusted": False,
        })

    # --- KB-generated: attach source chunk for grounded validation ---
    for i in range(k_start, k_end):
        pos = i - k_start
        chunk_idx = pos // config.KB_QUESTIONS_PER_CHUNK
        q_in_chunk = pos % config.KB_QUESTIONS_PER_CHUNK
        meta = kb_chunks[chunk_idx]
        enriched.append({
            "id": f"kb-{chunk_idx:04d}-q{q_in_chunk}",
            "question": records[i]["question"],
            "answer": records[i]["answer"],
            "source_type": "kb",
            "source_id": f"kb-{chunk_idx:04d}",
            "source_ref": {
                "kb_row_index": meta["kb_row_index"],
                "kb_company": meta["kb_company"],
                "kb_chunk": meta["chunk"],
            },
            "reference_answer": None,
            "trusted": False,
        })

    # --- Adversarial refusals ---
    for offset, i in enumerate(range(a_start, a_end)):
        enriched.append({
            "id": f"adv-{offset:04d}",
            "question": records[i]["question"],
            "answer": records[i]["answer"],
            "source_type": "adversarial",
            "source_id": f"adv-{offset:04d}",
            "source_ref": {},
            "reference_answer": None,
            "trusted": False,
        })

    with open(output_path, "w") as f:
        for rec in enriched:
            f.write(json.dumps(rec) + "\n")

    summary = {
        "input_total": total,
        "human": human_n,
        "paraphrase_kept": sum(1 for r in enriched if r["source_type"] == "paraphrase"),
        "paraphrase_dropped": len(dropped_paraphrases),
        "kb": kb_n,
        "adversarial": adversarial_n,
        "enriched_total": len(enriched),
        "output": str(output_path),
    }
    logger.info("Backfill summary: %s", json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill()
