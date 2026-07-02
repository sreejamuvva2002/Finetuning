"""Data augmentation for fine-tuning - generates synthetic Q&A pairs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from . import config
    from .llm_client import LLMClient, get_client
except ImportError:
    import config
    from llm_client import LLMClient, get_client

logger = logging.getLogger(__name__)


def _parse_json_array(response: str) -> list[dict]:
    """Parse a JSON array, tolerating an optional Markdown code fence."""
    text = response.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        closing_fence = text.rfind("```")
        if first_newline != -1 and closing_fence > first_newline:
            text = text[first_newline + 1 : closing_fence].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, received {type(data).__name__}")

    valid_items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            valid_items.append({"question": question, "answer": answer})
    return valid_items


def _generate_json_array(
    client: LLMClient,
    prompt: str,
    context: str,
    max_tokens: Optional[int] = None,
    attempts: int = 3,
) -> list[dict]:
    """Generate and parse a JSON array, retrying malformed responses."""
    for attempt in range(1, attempts + 1):
        try:
            response = client.generate(prompt, max_tokens=max_tokens)
            return _parse_json_array(response)
        except Exception as exc:
            if attempt == attempts:
                logger.error("%s failed after %d attempts: %s", context, attempts, exc)
            else:
                logger.warning(
                    "%s returned invalid output (attempt %d/%d): %s; retrying",
                    context,
                    attempt,
                    attempts,
                    exc,
                )
    return []


class QuestionParaphraser:
    """Generate paraphrases of validated questions."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()

    def paraphrase_single(
        self, question: str, answer: str, num_variations: int = 5
    ) -> list[dict]:
        """Generate paraphrases of a single question-answer pair.

        Args:
            question: Original question
            answer: Golden answer
            num_variations: Number of variations to generate

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        prompt = f"""Given this Q&A pair from an EV Intelligence knowledge base:

Question: {question}
Answer: {answer}

Generate {num_variations} natural variations of this question. Vary the tone, formality, and phrasing while keeping the core intent identical.

Format your response as a JSON array where each item has "question" and "answer" keys:
[
  {{"question": "variation 1", "answer": "answer"}},
  {{"question": "variation 2", "answer": "answer"}},
  ...
]

Only return valid JSON, no other text."""

        return _generate_json_array(
            self.client,
            prompt,
            context=f"Paraphrasing '{question[:50]}...'",
        )[:num_variations]

    def paraphrase_batch(
        self,
        questions: list[dict],
        num_variations_per_question: int = 5,
    ) -> list[dict]:
        """Paraphrase a batch of questions.

        Args:
            questions: List of dicts with 'question' and 'answer' keys
            num_variations_per_question: Variations per original question

        Returns:
            List of all paraphrased Q&A pairs
        """
        all_variations = []
        for i, qa in enumerate(questions):
            logger.info(
                f"Paraphrasing {i+1}/{len(questions)}: {qa['question'][:50]}..."
            )
            variations = self.paraphrase_single(
                qa["question"],
                qa["answer"],
                num_variations=num_variations_per_question,
            )
            all_variations.extend(variations)

        logger.info(f"Generated {len(all_variations)} paraphrased Q&A pairs")
        return all_variations


class KBQuestionGenerator:
    """Generate new questions grounded in Knowledge Base."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()
        self.kb_chunks = self._load_kb_chunks()

    def _load_kb_chunks(self) -> list[str]:
        """Load each row of the raw Knowledge Base as a grounded text chunk."""
        try:
            df = pd.read_excel(config.KB_DATA).dropna(how="all")
            if config.KB_RECORD_LIMIT > 0 and len(df) > config.KB_RECORD_LIMIT:
                df = df.sample(n=config.KB_RECORD_LIMIT, random_state=42)

            chunks = []
            for _, row in df.iterrows():
                fields = []
                for column, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        fields.append(f"{column}: {str(value).strip()}")
                if fields:
                    chunks.append(" | ".join(fields))
            logger.info(
                "Loaded %d raw KB records from %s (limit=%s)",
                len(chunks),
                config.KB_DATA,
                config.KB_RECORD_LIMIT or "all",
            )
            return chunks
        except Exception as e:
            logger.warning(f"Failed to load KB chunks: {e}. Using empty list.")
            return []

    def generate_from_chunk(self, chunk: str, num_questions: int = 3) -> list[dict]:
        """Generate Q&A pairs from a single KB chunk.

        Args:
            chunk: Text chunk from Knowledge Base
            num_questions: Number of questions to generate

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        # Truncate chunk if too long
        chunk = chunk[:1000]

        prompt = f"""You are an EV Intelligence expert. Based on the following knowledge base text:

{chunk}

Generate {num_questions} diverse, domain-specific questions that can be answered ONLY using this text.
For each question, provide a detailed answer grounded entirely in the provided text.

Important:
- Questions should be realistic and challenging
- Answers must use only information from the text (no external knowledge)
- Vary question types (what, how, why, when, where)

Format as JSON array with "question" and "answer" keys:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Return only valid JSON."""

        return _generate_json_array(
            self.client,
            prompt,
            context="KB question generation",
        )[:num_questions]

    def generate_batch(self, num_questions_per_chunk: int = 3) -> list[dict]:
        """Generate questions from all KB chunks.

        Args:
            num_questions_per_chunk: Questions per chunk

        Returns:
            List of all generated Q&A pairs
        """
        all_questions = []
        for i, chunk in enumerate(self.kb_chunks):
            logger.info(f"Generating from chunk {i+1}/{len(self.kb_chunks)}")
            questions = self.generate_from_chunk(chunk, num_questions=num_questions_per_chunk)
            all_questions.extend(questions)

        logger.info(f"Generated {len(all_questions)} KB-driven Q&A pairs")
        return all_questions

    def generate_adversarial(self, num_samples: int = 50) -> list[dict]:
        """Generate adversarial 'I don't know' questions.

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        all_questions = []
        batch_size = 10
        for batch_start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_start)
            prompt = f"""Generate {current_batch_size} realistic questions about Electric Vehicles and related topics
that CANNOT be answered using a knowledge base about EV Intelligence, charging infrastructure,
and logistics in Georgia.

For each, provide an answer that appropriately declines to answer, like:
"I don't have information about this topic in my knowledge base."

Format as JSON:
[
  {{"question": "...", "answer": "I don't have information about..."}},
  ...
]

Return only valid JSON."""

            logger.info(
                "Generating adversarial batch %d/%d",
                batch_start // batch_size + 1,
                (num_samples + batch_size - 1) // batch_size,
            )
            questions = _generate_json_array(
                self.client,
                prompt,
                context="Adversarial question generation",
                max_tokens=2048,
            )
            all_questions.extend(questions[:current_batch_size])

        return all_questions[:num_samples]


def load_validated_questions() -> list[dict]:
    """Load all human-validated questions from the configured workbook."""
    try:
        df = pd.read_excel(config.HUMAN_QA_EXCEL)
        columns = {
            str(column).strip().casefold(): column
            for column in df.columns
            if pd.notna(column)
        }

        def find_column(candidates: tuple[str, ...]):
            for candidate in candidates:
                if candidate.casefold() in columns:
                    return columns[candidate.casefold()]
            return None

        question_column = find_column(("Question", "Questions"))
        answer_column = find_column(
            ("Human validated answers", "Human validated answer", "Answer", "Answers")
        )
        if question_column is None or answer_column is None:
            raise ValueError(
                "Expected question and answer columns; found "
                f"{[str(column) for column in df.columns if pd.notna(column)]}"
            )

        questions = []
        for _, row in df.iterrows():
            question = row.get(question_column)
            answer = row.get(answer_column)
            if pd.isna(question) or pd.isna(answer):
                continue
            question = str(question).strip()
            answer = str(answer).strip()
            if not question or not answer:
                continue
            questions.append(
                {
                    "question": question,
                    "answer": answer,
                }
            )
        logger.info(f"Loaded {len(questions)} validated questions")
        return questions
    except Exception as e:
        logger.error(f"Failed to load validated questions: {e}")
        return []


def save_augmented_questions(questions: list[dict], output_path: Optional[Path] = None):
    """Save augmented questions to JSONL."""
    if output_path is None:
        output_path = config.AUGMENTED_QUESTIONS_JSONL

    with open(output_path, "w") as f:
        for qa in questions:
            f.write(json.dumps(qa) + "\n")

    logger.info(f"Saved {len(questions)} augmented questions to {output_path}")


def augment_dataset(
    paraphrase_count: int = config.PARAPHRASE_COUNT,
    kb_questions_per_chunk: int = config.KB_QUESTIONS_PER_CHUNK,
    include_adversarial: bool = config.INCLUDE_ADVERSARIAL,
) -> list[dict]:
    """Run full data augmentation pipeline.

    Returns:
        List of all augmented Q&A pairs
    """
    logger.info("Starting data augmentation pipeline...")

    # Load validated questions
    validated = load_validated_questions()
    if not validated:
        logger.error("No validated questions loaded. Aborting.")
        return []

    # Paraphrase validated questions
    logger.info(f"Paraphrasing {len(validated)} validated questions...")
    paraphraser = QuestionParaphraser()
    paraphrases = paraphraser.paraphrase_batch(
        validated, num_variations_per_question=paraphrase_count
    )

    # Generate KB-driven questions
    logger.info("Generating KB-driven questions...")
    kb_generator = KBQuestionGenerator()
    kb_questions = kb_generator.generate_batch(num_questions_per_chunk=kb_questions_per_chunk)

    # Generate adversarial questions
    adversarial = []
    if include_adversarial:
        logger.info("Generating adversarial questions...")
        adversarial = kb_generator.generate_adversarial(num_samples=50)

    # Combine all
    all_augmented = validated + paraphrases + kb_questions + adversarial
    logger.info(
        f"Total augmented dataset: {len(all_augmented)} Q&A pairs "
        f"({len(validated)} original + {len(paraphrases)} paraphrases + "
        f"{len(kb_questions)} KB-driven + {len(adversarial)} adversarial)"
    )

    # Save
    save_augmented_questions(all_augmented)

    return all_augmented


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    augment_dataset()
