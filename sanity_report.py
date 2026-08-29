"""Post-recovery sanity report for the validated fine-tuning dataset (no LLM calls).

Checks:
  1. Per-segment retention (how many of each source_type survived validation).
  2. All 42 trusted human originals are present.
  3. Adversarial refusals are retained (> 0).
  4. Group-aware split has NO source_id leakage across train/val.
  5. Source-data quality scan: flag implausible Employment outliers in the raw KB. These
     propagate into "faithful" answers (e.g. the WIKA USA 250,000 value that lives in the KB
     itself), so the fix belongs upstream in GNEM_Excel_Data.xlsx, not in validation.
  6. Deduplication effect.
"""
from __future__ import annotations

import json
from collections import Counter

import pandas as pd

try:
    from . import config
    from .dataset_formatter import deduplicate, train_val_split
except ImportError:
    import config
    from dataset_formatter import deduplicate, train_val_split

# Single Georgia facility headcounts above this are almost certainly data-entry errors.
EMPLOYMENT_OUTLIER_THRESHOLD = 50_000


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main() -> None:
    validated = _load(config.VALIDATED_QUESTIONS_JSONL)
    enriched = _load(config.AUGMENTED_ENRICHED_JSONL)

    print("=" * 70)
    print("SANITY REPORT")
    print("=" * 70)

    # 1. Per-segment retention
    in_counts = Counter(r["source_type"] for r in enriched)
    out_counts = Counter(r.get("source_type") for r in validated)
    print("\n[1] Per-segment retention (validated / enriched input):")
    for seg in ("human", "paraphrase", "kb", "adversarial"):
        print(f"    {seg:12s}: {out_counts.get(seg,0):4d} / {in_counts.get(seg,0):4d}")
    print(f"    {'TOTAL':12s}: {len(validated):4d} / {len(enriched):4d}")

    # 2. All human originals retained
    human_ids = {r["id"] for r in enriched if r["source_type"] == "human"}
    kept_human = {r.get("id") for r in validated if r.get("source_type") == "human"}
    missing = human_ids - kept_human
    print(f"\n[2] Human originals retained: {len(kept_human)}/{len(human_ids)} "
          f"-> {'PASS' if not missing else 'FAIL: missing ' + str(missing)}")

    # 3. Refusals retained
    n_ref = out_counts.get("adversarial", 0)
    print(f"\n[3] Adversarial refusals retained: {n_ref} -> "
          f"{'PASS' if n_ref > 0 else 'FAIL'}")

    # 4. Split leakage (reproduce the exact split the formatter uses)
    deduped = deduplicate(list(validated))
    train, val = train_val_split(deduped, train_ratio=0.8)
    train_ids = {r.get("source_id") for r in train} - {None}
    val_ids = {r.get("source_id") for r in val} - {None}
    leak = train_ids & val_ids
    print(f"\n[4] Train/val split: {len(train)} train / {len(val)} val")
    print(f"    source_id leakage across splits: {len(leak)} -> "
          f"{'PASS' if not leak else 'FAIL: ' + str(list(leak)[:5])}")

    # 5. Source-data quality scan (raw KB). Grounded validation trusts the KB row, so any
    #    implausible values in the KB flow through as "faithful" answers. Surface them here so
    #    they can be fixed at the source rather than silently trained on.
    df = pd.read_excel(config.KB_DATA)
    emp = pd.to_numeric(df["Employment"], errors="coerce")
    outliers = df.loc[emp > EMPLOYMENT_OUTLIER_THRESHOLD, ["Company", "Location", "Employment"]]
    print(f"\n[5] Raw KB Employment outliers (> {EMPLOYMENT_OUTLIER_THRESHOLD:,}): "
          f"{len(outliers)} flagged for source review")
    for _, r in outliers.iterrows():
        print(f"    - {r['Company']} ({r['Location']}): {r['Employment']:,}")
    print("    NOTE: these are source-data issues in GNEM_Excel_Data.xlsx, not validation bugs.")

    # 6. Dedup effect
    print(f"\n[6] Deduplication: {len(validated)} -> {len(deduped)} "
          f"({len(validated) - len(deduped)} duplicate questions removed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
