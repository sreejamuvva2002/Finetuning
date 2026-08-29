"""Quarantine training examples that depend on confirmed KB source-data errors.

Eight rows in GNEM_Excel_Data.xlsx carry implausible Employment values (> 50,000 for a
single Georgia facility). We do NOT edit the workbook and we do NOT guess replacements.
Instead we temporarily quarantine every example whose correctness depends on those values -
including trusted human answers, because trusted status must not override a confirmed
source-data error - and regenerate the group-aware train/val split from what remains.

Quarantine rules (deterministic, auditable):
  * KB example (tied to an outlier row via kb_row_index) -> quarantined iff it is
    employment-dependent: an employment keyword appears in the question/answer, OR the
    answer contains one of the outlier values. Non-employment questions (role, location,
    classification, product) about the same company are KEPT - they are unaffected.
  * human / paraphrase example -> quarantined iff its answer is employment-context AND
    (contains an exact outlier value, OR names an outlier company, OR contains any number
    > 50,000). The last clause catches aggregates (e.g. a county total dominated by the
    erroneous rows) that don't literally repeat an outlier value.
  * Any example, any segment, whose answer contains an exact outlier value -> quarantined
    (catch-all).
Answers are identical within a source_id group, so groups quarantine coherently.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict, OrderedDict

import pandas as pd

try:
    from . import config
    from .dataset_formatter import format_validated_dataset
except ImportError:
    import config
    from dataset_formatter import format_validated_dataset

OUTLIER_THRESHOLD = 50_000
EMP_KW = re.compile(
    r"employ|headcount|workforce|staff|largest employer|highest employ|most employ|"
    r"total employ|number of (people|workers)|how many (people|workers)",
    re.I,
)

ACTIVE_JSONL = config.FINETUNING_OUTPUTS_DIR / "validated_active.jsonl"
QUARANTINE_JSONL = config.FINETUNING_OUTPUTS_DIR / "quarantined_examples.jsonl"
REPORT_JSON = config.FINETUNING_OUTPUTS_DIR / "quarantine_report.json"
REPORT_MD = config.ROOT / "DATA_QUALITY_REPORT.md"


def _nums(text: str) -> set[int]:
    return {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]{3,}", text)}


def load_outliers() -> list[dict]:
    """Return the outlier KB rows in pipeline row order, with their kb source_id."""
    df = pd.read_excel(config.KB_DATA).dropna(how="all").reset_index(drop=True)
    emp = pd.to_numeric(df["Employment"], errors="coerce")
    outliers = []
    for i in range(len(df)):
        if pd.notna(emp[i]) and emp[i] > OUTLIER_THRESHOLD:
            outliers.append({
                "kb_row_index": i,
                "source_id": f"kb-{i:04d}",
                "company": str(df.iloc[i]["Company"]).strip(),
                "location": str(df.iloc[i]["Location"]).strip(),
                "employment": int(emp[i]),
            })
    return outliers


def is_employment_context(rec: dict) -> bool:
    return bool(EMP_KW.search(rec["question"]) or EMP_KW.search(rec["answer"]))


def quarantine():
    validated = [json.loads(l) for l in open(config.VALIDATED_QUESTIONS_JSONL) if l.strip()]
    outliers = load_outliers()
    outlier_rows = {o["kb_row_index"] for o in outliers}
    outlier_vals = {o["employment"] for o in outliers}
    outlier_companies = sorted({o["company"] for o in outliers})

    def names_outlier_company(text: str) -> list[str]:
        low = text.lower()
        return [c for c in outlier_companies
                if c.lower() in low or c.lower().split(" ")[0] in low.split()]

    def should_quarantine(rec: dict) -> tuple[bool, str]:
        ans = rec["answer"]
        ans_nums = _nums(ans)
        # Catch-all: exact outlier value anywhere.
        if ans_nums & outlier_vals:
            return True, f"answer contains outlier value {sorted(ans_nums & outlier_vals)}"
        stype = rec.get("source_type")
        if stype == "kb":
            row = (rec.get("source_ref") or {}).get("kb_row_index")
            if row in outlier_rows and is_employment_context(rec):
                return True, f"employment question on outlier KB row {row}"
            return False, ""
        if stype in ("human", "paraphrase"):
            if is_employment_context(rec):
                comp = names_outlier_company(ans)
                if comp:
                    return True, f"employment answer names outlier company {comp}"
                big = sorted(n for n in ans_nums if n > OUTLIER_THRESHOLD)
                if big:
                    return True, f"employment aggregate with implausible total {big}"
            return False, ""
        return False, ""

    kept, removed = [], []
    reasons = {}
    for rec in validated:
        q, why = should_quarantine(rec)
        if q:
            removed.append(rec)
            reasons[rec.get("id")] = why
        else:
            kept.append(rec)

    # --- persist active / quarantined data (originals untouched) ---
    with open(ACTIVE_JSONL, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    with open(QUARANTINE_JSONL, "w") as f:
        for r in removed:
            f.write(json.dumps({**r, "quarantine_reason": reasons.get(r.get("id"))}) + "\n")

    # --- regenerate group-aware split from the active set only ---
    train_path, val_path = format_validated_dataset(input_path=ACTIVE_JSONL)

    # --- verify zero leakage on the regenerated split source ---
    from dataset_formatter import deduplicate, train_val_split  # noqa
    deduped = deduplicate(list(kept))
    train, val = train_val_split(deduped, train_ratio=0.8)
    train_ids = {r.get("source_id") for r in train} - {None}
    val_ids = {r.get("source_id") for r in val} - {None}
    leakage = sorted(train_ids & val_ids)

    # --- build structured report ---
    removed_by_seg = defaultdict(int)
    for r in removed:
        removed_by_seg[r.get("source_type")] += 1

    # map derived examples to each outlier company
    def derived_for(company: str) -> dict:
        first = company.lower().split(" ")[0]
        kb_ids = [r.get("id") for r in validated if r.get("source_type") == "kb"
                  and (r.get("source_ref") or {}).get("kb_row_index") in
                  {o["kb_row_index"] for o in outliers if o["company"] == company}]
        hp_ids = [r.get("id") for r in validated if r.get("source_type") in ("human", "paraphrase")
                  and (first in r["answer"].lower())]
        return {"kb_examples": kb_ids, "human_paraphrase_examples": sorted(hp_ids)}

    report = OrderedDict()
    report["outlier_rows"] = [
        {**o, "employment_flag": "IMPLAUSIBLE (> %d)" % OUTLIER_THRESHOLD,
         "derived_examples": derived_for(o["company"])}
        for o in outliers
    ]
    report["quarantine"] = {
        "rows_quarantined": len(outliers),
        "examples_removed_total": len(removed),
        "examples_removed_by_segment": dict(removed_by_seg),
        "removed_ids": {r.get("id"): reasons.get(r.get("id")) for r in removed},
    }
    report["result"] = {
        "validated_input": len(validated),
        "active_kept": len(kept),
        "train_count": len(train),
        "val_count": len(val),
        "source_id_leakage": leakage,
        "train_dataset": str(train_path),
        "val_dataset": str(val_path),
    }
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    _write_markdown(report, outliers, removed, reasons)
    return report


def _write_markdown(report, outliers, removed, reasons):
    lines = []
    lines.append("# Data-Quality Report — KB Employment Outliers\n")
    lines.append("Generated by `quarantine_outliers.py`. **The raw workbook "
                 "`kb/GNEM_Excel_Data.xlsx` was NOT modified.** No replacement values were "
                 "guessed. Affected examples are *temporarily quarantined* pending "
                 "authoritative corrected values.\n")
    lines.append("## 1. Outlier rows (Employment > 50,000 for a single Georgia facility)\n")
    lines.append("| KB row | source_id | Company | Location | Employment (raw) |")
    lines.append("|---|---|---|---|---|")
    for o in outliers:
        lines.append(f"| {o['kb_row_index']} | `{o['source_id']}` | {o['company']} | "
                     f"{o['location']} | **{o['employment']:,}** |")
    lines.append("\n## 2. Examples derived from each outlier company\n")
    for entry in report["outlier_rows"]:
        d = entry["derived_examples"]
        lines.append(f"### {entry['company']} (`{entry['source_id']}`, "
                     f"raw employment {entry['employment']:,})")
        lines.append(f"- KB examples from this row: "
                     f"{', '.join('`'+i+'`' for i in d['kb_examples']) or '—'}")
        lines.append(f"- Human/paraphrase examples referencing it: "
                     f"{', '.join('`'+i+'`' for i in d['human_paraphrase_examples']) or '—'}\n")
    lines.append("## 3. Quarantined examples (removed from train/val)\n")
    lines.append("| id | source_type | reason |")
    lines.append("|---|---|---|")
    for r in removed:
        lines.append(f"| `{r.get('id')}` | {r.get('source_type')} | "
                     f"{reasons.get(r.get('id'))} |")
    lines.append("\n> Trusted human answers are included above where they depend on a "
                 "confirmed source error (trusted status does not override it).\n")
    q = report["quarantine"]
    res = report["result"]
    lines.append("## 4. Summary\n")
    lines.append(f"- **Outlier rows quarantined:** {q['rows_quarantined']}")
    lines.append(f"- **Examples removed (total):** {q['examples_removed_total']}")
    for seg, n in sorted(q["examples_removed_by_segment"].items()):
        lines.append(f"  - {seg}: {n}")
    lines.append(f"- **Validated input:** {res['validated_input']} → "
                 f"**active kept:** {res['active_kept']}")
    lines.append(f"- **Final train / val:** {res['train_count']} / {res['val_count']}")
    lines.append(f"- **source_id leakage across splits:** "
                 f"{len(res['source_id_leakage'])} "
                 f"({'PASS' if not res['source_id_leakage'] else 'FAIL'})")
    lines.append("\n## 5. Non-employment questions retained\n")
    lines.append("Role / location / classification / product questions about the same "
                 "companies are **kept**, since they do not depend on the erroneous "
                 "employment values.\n")
    REPORT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    rep = quarantine()
    print(json.dumps({"quarantine": rep["quarantine"]["examples_removed_by_segment"],
                      "result": rep["result"]}, indent=2))
