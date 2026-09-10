"""Automatic evaluation metrics for Phase 6 (build spec section 8).

chrF++ is primary; BLEU/ROUGE-L/BERTScore/AfriCOMET are secondary; diacritic
accuracy and language consistency are the two task-specific measures.
Significance is paired bootstrap resampling against system M.

Heavy optional dependencies (bert_score, comet) are imported lazily so that
`import src.eval.metrics` never requires a GPU or a multi-GB model download;
callers that need those metrics get a clear None + must log why it was
skipped (see scripts/10_evaluate.py), never a silently fabricated number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.yoruba_text import validate_charset

# --- chrF++ / BLEU / ROUGE-L -------------------------------------------------

def corpus_chrf(hyps: list[str], refs: list[str], word_order: int = 2) -> float:
    """chrF++ (word_order=2) corpus score, Post (2018) / Popovic (2017)."""
    import sacrebleu

    return sacrebleu.corpus_chrf(hyps, [refs], word_order=word_order).score


def sentence_chrf_scores(hyps: list[str], refs: list[str], word_order: int = 2) -> list[float]:
    """Per-sentence chrF++ scores, aligned 1:1 with hyps/refs. Feed this into
    paired_bootstrap for significance testing."""
    import sacrebleu

    return [sacrebleu.sentence_chrf(h, [r], word_order=word_order).score for h, r in zip(hyps, refs)]


def corpus_bleu(hyps: list[str], refs: list[str]) -> tuple[float, str]:
    """sacreBLEU corpus score plus its reproducibility signature (Post, 2018).

    The signature must come from the BLEU metric object (`.get_signature()`),
    not the score result -- newer sacrebleu versions no longer expose it on
    the returned BLEUScore.
    """
    import sacrebleu

    bleu_metric = sacrebleu.BLEU()
    result = bleu_metric.corpus_score(hyps, [refs])
    return result.score, str(bleu_metric.get_signature())


def sentence_bleu_scores(hyps: list[str], refs: list[str]) -> list[float]:
    import sacrebleu

    return [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)]


def corpus_rouge_l(hyps: list[str], refs: list[str]) -> float:
    """Mean per-example ROUGE-L F-measure."""
    scores = sentence_rouge_l_scores(hyps, refs)
    return sum(scores) / len(scores) if scores else 0.0


def sentence_rouge_l_scores(hyps: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return [scorer.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]


# --- BERTScore / AfriCOMET (optional, heavy) --------------------------------

def corpus_bertscore(hyps: list[str], refs: list[str], lang: str = "yo") -> float | None:
    """Mean BERTScore F1, or None if the `bert-score` package / model weights
    are unavailable. Callers MUST record a skip reason rather than omit the
    column silently -- see scripts/10_evaluate.py."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return None
    _, _, f1 = bert_score_fn(hyps, refs, lang=lang, rescale_with_baseline=False)
    return float(f1.mean())


def corpus_africomet(hyps: list[str], refs: list[str], sources: list[str], checkpoint_path: str | None) -> float | None:
    """AfriCOMET score, or None if no checkpoint is configured
    (configs/eval.yaml: africomet.checkpoint_path)."""
    if not checkpoint_path:
        return None
    from comet import load_from_checkpoint

    model = load_from_checkpoint(checkpoint_path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    output = model.predict(data, batch_size=8, gpus=1 if _cuda_available() else 0)
    return float(output.system_score)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# --- task-specific: language consistency ------------------------------------

# Below this proportion of Yoruba-consistent tokens, treat a Yoruba text field
# as "looks untranslated/mostly-English" rather than "a normal sentence with a
# few embedded proper nouns/technical terms". Calibrated against real
# reviewer-corrected text (postedit_sample_200.xlsx): the single worst
# legitimate case -- a short sentence naming COVID-19, coronavirus, and
# SARS-CoV-2 -- scored 0.625; genuinely untranslated English scores near 0.
# Shared by scripts/04b_import_postedit.py and src/data/schema.py so the
# threshold and its justification live in exactly one place.
MIN_LANGUAGE_CONSISTENCY_FOR_YORUBA_TEXT = 0.5


def language_consistency(hyps: list[str]) -> float:
    """Proportion of whitespace tokens across `hyps` that use only characters
    permitted in Yoruba orthography (strict charset check per token).

    This is an orthographic proxy, not a trained language-ID classifier: a
    short English function word built entirely from the shared Latin subset
    (e.g. "an", "in") is indistinguishable from Yoruba by this test. That is
    an accepted limitation -- the failure mode this metric targets is visible
    English code-switching / English sentences in the output, which reliably
    contain c/q/v/x/z, apostrophically-contracted words, or other characters
    outside the strict Yoruba set and so are correctly flagged.
    """
    total = 0
    consistent = 0
    for text in hyps:
        for token in text.split():
            token_clean = token.strip(".,;:!?'\"()")
            if not token_clean:
                continue
            total += 1
            if validate_charset(token_clean, strict=True).is_valid:
                consistent += 1
    return consistent / total if total else 0.0


# --- significance: paired bootstrap ------------------------------------------

@dataclass
class BootstrapResult:
    mean_diff: float       # mean(scores_m) - mean(scores_baseline); positive => M better
    ci_low: float
    ci_high: float
    p_value: float          # two-sided
    effect_size: float      # Cohen's d on the paired per-example differences
    n_resamples: int


def paired_bootstrap(scores_baseline: list[float], scores_m: list[float], n_resamples: int = 1000,
                      ci: float = 0.95, seed: int = 42) -> BootstrapResult:
    """Paired bootstrap resampling (Efron & Tibshirani, 1993 style) comparing
    a baseline system's per-example metric scores against M's, on the SAME
    ordered set of test items (i.e. scores_baseline[i] and scores_m[i] must be
    the same test example). Use sentence_chrf_scores / sentence_bleu_scores /
    sentence_rouge_l_scores to produce the per-example inputs.
    """
    if len(scores_baseline) != len(scores_m):
        raise ValueError("paired_bootstrap requires equal-length, aligned score lists")
    n = len(scores_baseline)
    if n == 0:
        raise ValueError("paired_bootstrap requires at least one paired example")

    rng = np.random.default_rng(seed)
    a = np.asarray(scores_baseline, dtype=float)
    b = np.asarray(scores_m, dtype=float)
    diffs = b - a
    observed = float(diffs.mean())

    resampled_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_means[i] = diffs[idx].mean()

    alpha = 1 - ci
    lo = float(np.quantile(resampled_means, alpha / 2))
    hi = float(np.quantile(resampled_means, 1 - alpha / 2))

    # two-sided p-value: proportion of resamples on the opposite side of zero
    # from the observed effect, doubled (Davison & Hinkley, 1997, ch.4-style
    # bootstrap hypothesis test).
    if observed >= 0:
        p_value = float(np.mean(resampled_means <= 0)) * 2
    else:
        p_value = float(np.mean(resampled_means >= 0)) * 2
    p_value = min(1.0, p_value)

    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    effect_size = observed / sd if sd > 0 else 0.0

    return BootstrapResult(
        mean_diff=observed, ci_low=lo, ci_high=hi, p_value=p_value,
        effect_size=effect_size, n_resamples=n_resamples,
    )
