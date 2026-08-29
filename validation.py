"""Validation pipeline - grounded LLM-as-judge scoring for synthetic data quality.

Key properties (fixing the earlier inverted acceptance):
  * ACCURACY is judged against the *source of truth* - the originating KB row for KB pairs,
    or the trusted reference answer for paraphrases - so correct answers stop being penalized
    for being "unverifiable".
  * Acceptance requires EVERY dimension to meet its threshold, never an average, so a
    wrong-but-fluent answer can't be rescued by high relevance/completeness.
  * Trusted human originals BYPASS the judge entirely.
  * Adversarial refusals are graded on appropriateness-of-decline, not completeness.
  * Routing is driven by each record's ``source_type`` provenance.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from . import config
    from .llm_client import LLMClient, get_client
except ImportError:
    import config
    from llm_client import LLMClient, get_client

logger = logging.getLogger(__name__)


def _extract_json(text: str):
    """Parse a JSON object or array, tolerating markdown fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        fence = text.rfind("```")
        if nl != -1 and fence > nl:
            text = text[nl + 1:fence].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the widest {...} or [...] span.
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No parseable JSON in response: {text[:200]!r}")


def _clamp_scores(data: dict) -> dict:
    """Coerce a raw judge dict into integer 1-5 scores plus comments."""
    def as_int(v, default=0):
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return default
    return {
        "accuracy": as_int(data.get("accuracy")),
        "completeness": as_int(data.get("completeness")),
        "relevance": as_int(data.get("relevance")),
        "comments": str(data.get("comments", "")),
    }


class QAValidator:
    """Grounded LLM-as-judge scorer with provenance-aware routing."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()

    # ---- acceptance policy ------------------------------------------------
    @staticmethod
    def accept(scores: dict) -> bool:
        """Accept only when EVERY dimension independently clears its threshold."""
        return (
            scores["accuracy"] >= config.MIN_ACCURACY
            and scores["completeness"] >= config.MIN_COMPLETENESS
            and scores["relevance"] >= config.MIN_RELEVANCE
        )

    # ---- single grounded QA scoring --------------------------------------
    def score_qa_grounded(self, question: str, answer: str, context: Optional[str]) -> dict:
        """Score one Q&A pair; ``context`` (KB row / reference answer) is the ground truth."""
        if context:
            grounding = f"""GROUND TRUTH (authoritative source for this Q&A):
{context}

Judge ACCURACY strictly against this ground truth. If the answer is consistent with the
ground truth, accuracy is high (5). Do NOT lower accuracy merely because you cannot
independently verify the facts elsewhere - the ground truth above IS authoritative. Lower
accuracy only when the answer contradicts the ground truth or invents facts not supported
by it."""
        else:
            grounding = ("No external ground truth is provided; judge accuracy on internal "
                         "consistency and domain plausibility.")

        prompt = f"""You are an expert evaluator for a Georgia EV Intelligence Q&A system.
{grounding}

Score the question-answer pair on three 1-5 dimensions:
1. ACCURACY: correctness against the ground truth (1=contradicts/invented, 5=fully supported)
2. COMPLETENESS: does the answer fully address the question (1=missing key details, 5=comprehensive)
3. RELEVANCE: is the question on-topic for EV intelligence/supply-chain/logistics (1=off-topic, 5=highly relevant)

Question: {question}
Answer: {answer}

Respond with JSON only:
{{"accuracy": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "comments": "<brief>"}}"""

        try:
            data = _extract_json(self.client.generate(prompt, max_tokens=1000))
            return _clamp_scores(data)
        except Exception as e:  # noqa: BLE001
            logger.error("Grounded scoring failed: %s", e)
            return {"accuracy": 0, "completeness": 0, "relevance": 0, "comments": f"error: {e}"}

    # ---- batched KB scoring (one call per KB row) ------------------------
    def score_kb_group(self, chunk: str, qa_list: list[dict]) -> list[dict]:
        """Score every Q&A generated from one KB row in a single grounded call.

        Falls back to per-item grounded scoring if the batched response can't be aligned.
        """
        numbered = "\n".join(
            f'[{i}] Q: {qa["question"]}\n    A: {qa["answer"]}' for i, qa in enumerate(qa_list)
        )
        prompt = f"""You are an expert evaluator for a Georgia EV Intelligence Q&A system.

GROUND TRUTH (the single knowledge-base row these Q&A pairs were generated from):
{chunk}

This row is authoritative. Judge each answer's ACCURACY strictly against it: an answer that
is consistent with the row is accurate (5); do NOT penalize for being unverifiable elsewhere.
Penalize only contradictions or invented facts not supported by the row.

Score each item on ACCURACY, COMPLETENESS, RELEVANCE (each 1-5):
{numbered}

Respond with a JSON array, one object per item, in the same order:
[{{"index": 0, "accuracy": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "comments": "<brief>"}}, ...]
Return only the JSON array."""

        try:
            # Budget must cover reasoning + one verbose object per item, or the JSON array
            # truncates and we fall back to (correct but slower) per-item grounded scoring.
            data = _extract_json(self.client.generate(prompt, max_tokens=3000))
            if isinstance(data, list):
                by_index = {}
                for item in data:
                    if isinstance(item, dict) and "index" in item:
                        by_index[int(item["index"])] = _clamp_scores(item)
                results = []
                for i in range(len(qa_list)):
                    if i in by_index:
                        results.append(by_index[i])
                    else:  # missing item -> ground it individually
                        results.append(self.score_qa_grounded(
                            qa_list[i]["question"], qa_list[i]["answer"], chunk))
                return results
        except Exception as e:  # noqa: BLE001
            logger.warning("Batched KB scoring failed (%s); scoring per-item", e)

        return [self.score_qa_grounded(qa["question"], qa["answer"], chunk) for qa in qa_list]

    # ---- paraphrase intent verification (one call per original) ----------
    def verify_paraphrase_group(self, original_q: str, variants: list[str]) -> list[bool]:
        """Return, per variant, whether it preserves the original question's intent."""
        numbered = "\n".join(f"[{i}] {v}" for i, v in enumerate(variants))
        prompt = f"""A knowledge-base question was paraphrased into variants. A variant is VALID
only if it asks for the same information as the original (same intent), even if worded
differently. It is INVALID if it changes the subject, scope, or the information requested.

ORIGINAL: {original_q}

VARIANTS:
{numbered}

Respond with JSON only: {{"valid_indices": [list of indices that preserve intent]}}"""
        try:
            # gpt-oss is a reasoning model; give it enough budget so reasoning tokens don't
            # starve the visible JSON answer.
            data = _extract_json(self.client.generate(prompt, max_tokens=1200))
            valid = set(int(i) for i in data.get("valid_indices", []))
            return [i in valid for i in range(len(variants))]
        except Exception as e:  # noqa: BLE001
            logger.warning("Paraphrase intent check failed (%s); keeping all variants", e)
            return [True] * len(variants)

    # ---- adversarial refusal scoring (batched) ---------------------------
    def score_refusal_batch(self, qa_list: list[dict]) -> list[dict]:
        """Grade refusals on appropriateness-of-decline rather than completeness."""
        numbered = "\n".join(
            f'[{i}] Q: {qa["question"]}\n    A: {qa["answer"]}' for i, qa in enumerate(qa_list)
        )
        prompt = f"""You are evaluating REFUSAL training examples for a Georgia EV Intelligence
assistant. Each answer SHOULD decline, because the question cannot be answered from a
knowledge base about Georgia EV supply chain, charging infrastructure, and logistics.

Score each item (1-5):
1. ACCURACY: is declining the correct behavior for this question? (5 = genuinely out-of-scope
   and correctly declined; 1 = actually answerable from a Georgia EV KB, so should NOT refuse)
2. COMPLETENESS: does the answer clearly and politely decline? (5 = clean refusal; 1 = confused)
3. RELEVANCE: is this a realistic question a user might plausibly ask? (5 = realistic)

Items:
{numbered}

Respond with a JSON array in order:
[{{"index": 0, "accuracy": <1-5>, "completeness": <1-5>, "relevance": <1-5>, "comments": "<brief>"}}, ...]
Return only the JSON array."""
        try:
            data = _extract_json(self.client.generate(prompt, max_tokens=1500))
            if isinstance(data, list):
                by_index = {int(it["index"]): _clamp_scores(it)
                            for it in data if isinstance(it, dict) and "index" in it}
                return [by_index.get(i, {"accuracy": 0, "completeness": 0,
                                         "relevance": 0, "comments": "missing"})
                        for i in range(len(qa_list))]
        except Exception as e:  # noqa: BLE001
            logger.warning("Batched refusal scoring failed: %s", e)
        return [{"accuracy": 0, "completeness": 0, "relevance": 0, "comments": "scoring error"}
                for _ in qa_list]


# ============================================================================
# Orchestration over the enriched (provenance-carrying) dataset
# ============================================================================

def _load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _finalize(rec: dict, decision: str, scores: Optional[dict], rubric: str) -> dict:
    """Attach a validation block, dropping the bulky kb_chunk from the persisted record."""
    ref = {k: v for k, v in (rec.get("source_ref") or {}).items() if k != "kb_chunk"}
    out = {
        "id": rec.get("id"),
        "question": rec["question"],
        "answer": rec["answer"],
        "source_type": rec.get("source_type"),
        "source_id": rec.get("source_id"),
        "source_ref": ref,
        "validation": {
            "decision": decision,
            "rubric": rubric,
            "grounded": rubric in ("qa", "refusal") and bool(scores),
            "judge_model": config.DATA_GEN_OLLAMA_MODEL,
            **({"scores": scores} if scores else {}),
        },
    }
    return out


def validate_enriched_dataset(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[list[dict], list[dict]]:
    """Route records by provenance and produce accepted/rejected sets.

    * human       -> bypass judge (trusted)
    * paraphrase  -> trusted answer + batched intent check
    * kb          -> grounded, one batched call per KB row
    * adversarial -> refusal rubric, batched
    """
    input_path = input_path or config.AUGMENTED_ENRICHED_JSONL
    output_dir = output_dir or config.FINETUNING_OUTPUTS_DIR
    records = _load_records(input_path)
    logger.info("Loaded %d enriched records from %s", len(records), input_path)

    validator = QAValidator()
    passed: list[dict] = []
    failed: list[dict] = []
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r.get("source_type", "unknown")].append(r)

    # --- human: bypass ---
    for r in by_type.get("human", []):
        passed.append(_finalize(r, "accept", None, "bypass"))
    logger.info("Human originals accepted (bypass): %d", len(by_type.get("human", [])))

    # --- paraphrase: batched intent check per parent (answer already trusted) ---
    para_by_parent: dict[str, list[dict]] = defaultdict(list)
    for r in by_type.get("paraphrase", []):
        para_by_parent[r["source_id"]].append(r)
    p_pass = p_fail = 0
    for parent_id, variants in para_by_parent.items():
        # The original question is not stored on the paraphrase; look it up from the human record.
        original_q = _original_question_for(records, parent_id) or variants[0]["question"]
        keep = validator.verify_paraphrase_group(original_q, [v["question"] for v in variants])
        for v, ok in zip(variants, keep):
            if ok:
                passed.append(_finalize(v, "accept", None, "bypass"))
                p_pass += 1
            else:
                failed.append(_finalize(v, "reject_intent", None, "bypass"))
                p_fail += 1
    logger.info("Paraphrases: %d kept, %d dropped (intent)", p_pass, p_fail)

    # --- kb: grounded, batched per source_id (== per KB row) ---
    kb_by_group: dict[str, list[dict]] = defaultdict(list)
    for r in by_type.get("kb", []):
        kb_by_group[r["source_id"]].append(r)
    kb_pass = kb_fail = 0
    for group in kb_by_group.values():
        chunk = (group[0].get("source_ref") or {}).get("kb_chunk", "")
        scores_list = validator.score_kb_group(chunk, group)
        for rec, scores in zip(group, scores_list):
            if validator.accept(scores):
                passed.append(_finalize(rec, "accept", scores, "qa"))
                kb_pass += 1
            else:
                failed.append(_finalize(rec, "reject", scores, "qa"))
                kb_fail += 1
    logger.info("KB pairs: %d passed, %d failed (grounded)", kb_pass, kb_fail)

    # --- adversarial: refusal rubric, batched in 10s ---
    adv = by_type.get("adversarial", [])
    a_pass = a_fail = 0
    for start in range(0, len(adv), 10):
        batch = adv[start:start + 10]
        scores_list = validator.score_refusal_batch(batch)
        for rec, scores in zip(batch, scores_list):
            if validator.accept(scores):
                passed.append(_finalize(rec, "accept", scores, "refusal"))
                a_pass += 1
            else:
                failed.append(_finalize(rec, "reject", scores, "refusal"))
                a_fail += 1
    logger.info("Adversarial refusals: %d passed, %d failed", a_pass, a_fail)

    _save(passed, failed, output_dir)
    return passed, failed


def _original_question_for(records: list[dict], parent_id: str) -> Optional[str]:
    for r in records:
        if r.get("id") == parent_id and r.get("source_type") == "human":
            return r["question"]
    return None


def _save(passed: list[dict], failed: list[dict], output_dir: Path) -> None:
    out = output_dir / "validated_questions.jsonl"
    with open(out, "w") as f:
        for rec in passed:
            f.write(json.dumps(rec) + "\n")
    logger.info("Saved %d validated records to %s", len(passed), out)

    def seg_counts(items):
        c: dict[str, int] = defaultdict(int)
        for it in items:
            c[it.get("source_type", "unknown")] += 1
        return dict(c)

    report = {
        "total": len(passed) + len(failed),
        "passed": len(passed),
        "failed": len(failed),
        "acceptance_rate": len(passed) / max(1, len(passed) + len(failed)),
        "passed_by_type": seg_counts(passed),
        "failed_by_type": seg_counts(failed),
        "failed_examples": failed[:10],
    }
    with open(output_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report: %s", json.dumps({k: report[k] for k in
                ("passed", "failed", "passed_by_type")}, indent=2))


def validate_augmented_dataset(
    input_path: Optional[Path] = None,
    min_score: int = config.VALIDATION_MIN_SCORE,
) -> tuple[list[dict], list[dict]]:
    """Entry point used by the CLI.

    If the input carries provenance (``source_type``), run the routed grounded validator.
    Otherwise fall back to flat grounded scoring for backward compatibility.
    """
    path = input_path or config.AUGMENTED_ENRICHED_JSONL
    if not Path(path).exists():
        logger.error("Input file not found: %s", path)
        return [], []

    records = _load_records(Path(path))
    if records and "source_type" in records[0]:
        return validate_enriched_dataset(Path(path))

    # Backward-compatible flat path (no provenance): grounded=None, per-dimension gate.
    logger.warning("Input has no provenance; running flat ungrounded scoring.")
    validator = QAValidator()
    passed, failed = [], []
    for qa in records:
        scores = validator.score_qa_grounded(qa["question"], qa["answer"], None)
        (passed if validator.accept(scores) else failed).append({**qa, "scores": scores})
    _save(passed, failed, config.FINETUNING_OUTPUTS_DIR)
    return passed, failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_augmented_dataset()
