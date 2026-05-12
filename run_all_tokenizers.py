#!/usr/bin/env python3
"""
Run all 10 tokenizers sequentially on the real historical corpus.
Loads corpus once, iterates per tokenizer with exception isolation.
Does NOT modify run_complete_pipeline.py or temporal_inference.py.
"""

import sys, os, time, json, math, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_complete_pipeline import load_corpus
from src.validation.temporal_inference import TemporalDistributionInference
from src.evaluation import run_evaluation
import types as _types

TOKENIZERS = [
    "bert-base-uncased",
    "gpt2",
    "roberta-base",
    "albert-base-v2",
    "distilbert-base-uncased",
    "google/electra-small-discriminator",
    "xlm-roberta-base",
    "t5-small",
    "distilgpt2",
    "gpt2-medium",
]

DATA_DIR   = "/Users/roshani/Downloads/MSc_AI_Project/data/real_historical_texts"
OUTPUT_DIR = Path("results/semantic_shift_run")
MAX_TEXTS  = 5
TIMEOUT_S  = 90 * 60  # 90 minutes total wall clock

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
failures_path = OUTPUT_DIR / "failures.log"


def _safe_ensemble(self_a, decade_patterns, methods=None, weights=None):
    """Heuristic-only ensemble — breaks mutual recursion with LP path."""
    try:
        filtered = self_a.remove_top_frequent_tokens(decade_patterns, 10)
        return self_a._infer_distribution_heuristic(filtered)
    except Exception as _e:
        decades_k = list(decade_patterns.keys())
        return {d: 1.0 / len(decades_k) for d in decades_k}


USABLE_SHIFT_WORDS = [
    'computer', 'program', 'record', 'play', 'drive',
    'tank', 'pilot', 'plastic', 'radiation', 'market', 'film'
]
USE_SEMANTIC = True
MIN_CONTEXTS = 30


def pretrain_classifiers(corpus: dict):
    """Train sense classifiers once — they are tokenizer-independent (sentence-transformers only).
    Returns (classifiers_dict, semantic_model) so both can be injected into each analyzer."""
    print("Pre-training sense classifiers (tokenizer-independent, runs once)...")
    t0 = time.time()
    trainer = TemporalDistributionInference("bert-base-uncased")
    trainer.shift_words = {k: v for k, v in trainer.shift_words.items()
                           if k in USABLE_SHIFT_WORDS}
    classifiers = trainer.train_sense_classifiers(corpus, min_contexts=MIN_CONTEXTS)
    semantic_model = getattr(trainer, "semantic_model", None)
    print(f"Pre-training done in {time.time()-t0:.1f}s  "
          f"Classifiers: {list(classifiers.keys())}")
    return classifiers, semantic_model


def run_one(tok_name: str, corpus: dict, pretrained_classifiers: dict, semantic_model) -> dict:
    # Ensure plots dir exists for tokenizer names that contain '/' (e.g. google/electra...)
    plot_dir = Path("results/temporal_inference") / Path(tok_name).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TemporalDistributionInference(tok_name)
    analyzer.infer_distribution_ensemble = _types.MethodType(_safe_ensemble, analyzer)

    if USE_SEMANTIC:
        analyzer.shift_words = {k: v for k, v in analyzer.shift_words.items()
                                if k in USABLE_SHIFT_WORDS}
        # Inject pre-trained classifiers AND semantic_model — skips re-training
        analyzer.sense_classifiers = pretrained_classifiers
        if semantic_model is not None:
            analyzer.semantic_model = semantic_model

    analysis = analyzer.run_analysis(
        corpus,
        use_semantic_shift=USE_SEMANTIC,
        min_contexts=MIN_CONTEXTS,
    )

    eval_list = run_evaluation(analyzer, corpus)
    trained = list(analyzer.sense_classifiers.keys()) if hasattr(analyzer, 'sense_classifiers') and analyzer.sense_classifiers else []

    return {
        "distribution":        analysis.get("distribution", {}),
        "evaluation":          eval_list,
        "classifiers_trained": trained,
    }


def peak_decade(dist: dict) -> str:
    if not dist:
        return "N/A"
    return max(dist, key=lambda d: dist[d])


print("Loading corpus (once)...")
t_corpus = time.time()
corpus = load_corpus(DATA_DIR, decades=None, max_per_decade=MAX_TEXTS)
print(f"Corpus loaded in {time.time()-t_corpus:.1f}s  "
      f"({sum(len(v) for v in corpus.values())} texts across {len(corpus)} decades)\n")

# Pre-train classifiers once — they are tokenizer-independent
pretrained, semantic_model = pretrain_classifiers(corpus) if USE_SEMANTIC else ({}, None)
print()

results_table = []   # (rank, tok, peak, mse, log10, elapsed)
distributions = {}   # tok -> dist dict
all_results   = {}   # tok -> full result dict

wall_start = time.time()

for idx, tok in enumerate(TOKENIZERS, 1):
    print(f"[{idx}/10] Starting: {tok}")
    t0 = time.time()
    try:
        if time.time() - wall_start > TIMEOUT_S:
            msg = f"TIMEOUT: global 30-minute limit reached before {tok}"
            print(msg)
            with open(failures_path, "a") as fh:
                fh.write(msg + "\n")
            break

        result = run_one(tok, corpus, pretrained, semantic_model)
        elapsed = time.time() - t0

        dist    = result.get("distribution", {})
        ev_list = result.get("evaluation") or []
        if isinstance(ev_list, list) and ev_list:
            avg_mse = sum(r["mse"] for r in ev_list) / len(ev_list)
        elif isinstance(ev_list, dict):
            avg_mse = ev_list.get("avg_mse", float("nan"))
        else:
            avg_mse = float("nan")
        log10 = math.log10(avg_mse) if avg_mse and avg_mse > 0 else float("nan")
        peak    = peak_decade(dist)

        print(f"[{idx}/10] {tok}: peak={peak}  mse={avg_mse:.6f}  "
              f"log10={log10:.4f}  time={elapsed:.1f}s")

        results_table.append((idx, tok, peak, avg_mse, log10, elapsed))
        distributions[tok] = dist
        all_results[tok]   = {**result, "_avg_mse": avg_mse}

        # Save per-tokenizer JSON
        safe_name = tok.replace("/", "_")
        out_path  = OUTPUT_DIR / f"{safe_name}.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"    Saved -> {out_path}\n")

    except Exception as exc:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        err_line = f"[{idx}/10] FAILED {tok} after {elapsed:.1f}s: {exc}"
        print(err_line)
        with open(failures_path, "a") as fh:
            fh.write(err_line + "\n")
            fh.write(tb + "\n\n")
        results_table.append((idx, tok, "FAILED", float("nan"), float("nan"), elapsed))
        print()

# ── RESULTS TABLE ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("RESULTS TABLE")
print("="*80)
print(f"{'#':<4} {'Tokenizer':<42} {'Peak':<10} {'MSE':<12} {'log10(MSE)':<12} {'Time(s)'}")
print("-"*80)
for row in results_table:
    idx, tok, peak, mse, log10, elapsed = row
    mse_s   = f"{mse:.6f}"  if not math.isnan(mse)   else "FAILED"
    log10_s = f"{log10:.4f}" if not math.isnan(log10) else "FAILED"
    print(f"{idx:<4} {tok:<42} {peak:<10} {mse_s:<12} {log10_s:<12} {elapsed:.1f}")

# ── DISTRIBUTION TABLE ───────────────────────────────────────────────────────
ALL_DECADES = [
    "1850s","1860s","1870s","1880s","1890s","1900s","1910s","1920s","1930s",
    "1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s","2020s",
]

print("\n" + "="*80)
print("DISTRIBUTION TABLE  (probability mass per decade)")
print("="*80)
header = f"{'Tokenizer':<42} " + " ".join(f"{d:<8}" for d in ALL_DECADES)
print(header)
print("-"*len(header))
for _, tok, peak, _, _, _ in results_table:
    dist = distributions.get(tok, {})
    if not dist:
        row_str = "  FAILED"
    else:
        row_str = " ".join(f"{dist.get(d, 0.0):.4f}  " for d in ALL_DECADES)
    print(f"{tok:<42} {row_str}")

# ── SUMMARY JSON ─────────────────────────────────────────────────────────────
summary = {
    "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "total_wall_seconds": round(time.time() - wall_start, 1),
    "tokenizers_attempted": len(TOKENIZERS),
    "tokenizers_succeeded": sum(1 for r in results_table if r[2] != "FAILED"),
    "per_tokenizer": [
        {
            "rank":       r[0],
            "tokenizer":  r[1],
            "peak_decade":r[2],
            "avg_mse":    None if math.isnan(r[3]) else round(r[3], 8),
            "log10_mse":  None if math.isnan(r[4]) else round(r[4], 4),
            "runtime_s":  round(r[5], 1),
            "distribution": distributions.get(r[1], {}),
        }
        for r in results_table
    ],
}

summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2, default=str)
print(f"\nSummary saved -> {summary_path}")
print(f"Total wall time: {summary['total_wall_seconds']}s")
