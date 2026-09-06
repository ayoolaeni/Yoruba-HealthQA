import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "adapt_script", Path(__file__).resolve().parents[1] / "scripts" / "08_adapt.py"
)
adapt_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapt_script)


def test_load_mixed_corpus_respects_ratio_when_both_corpora_are_large(tmp_path):
    yo_dir = tmp_path / "yo"
    en_dir = tmp_path / "en"
    yo_dir.mkdir()
    en_dir.mkdir()
    (yo_dir / "a.txt").write_text("\n".join(f"yoruba line {i}" for i in range(100)), encoding="utf-8")
    (en_dir / "a.txt").write_text("\n".join(f"english line {i}" for i in range(100)), encoding="utf-8")

    mixed = adapt_script.load_mixed_corpus(str(yo_dir / "*.txt"), str(en_dir / "*.txt"),
                                            mixture_ratio_yoruba=0.7, seed=42)
    n_yo = sum(1 for line in mixed if line.startswith("yoruba"))
    n_en = sum(1 for line in mixed if line.startswith("english"))
    assert n_yo + n_en == len(mixed)
    assert n_yo / len(mixed) == pytest.approx(0.7, abs=0.02)


def test_load_mixed_corpus_never_repeats_lines_beyond_one_pass(tmp_path):
    yo_dir = tmp_path / "yo"
    en_dir = tmp_path / "en"
    yo_dir.mkdir()
    en_dir.mkdir()
    (yo_dir / "a.txt").write_text("only one yoruba line", encoding="utf-8")
    (en_dir / "a.txt").write_text("\n".join(f"english line {i}" for i in range(50)), encoding="utf-8")

    # even asking for a high Yoruba ratio, the corpus only has 1 Yoruba line
    mixed = adapt_script.load_mixed_corpus(str(yo_dir / "*.txt"), str(en_dir / "*.txt"),
                                            mixture_ratio_yoruba=0.9, seed=1)
    assert mixed.count("only one yoruba line") == 1


def test_load_mixed_corpus_raises_on_missing_yoruba_corpus(tmp_path):
    en_dir = tmp_path / "en"
    en_dir.mkdir()
    (en_dir / "a.txt").write_text("english line", encoding="utf-8")

    with pytest.raises(Exception):
        adapt_script.load_mixed_corpus(str(tmp_path / "nonexistent" / "*.txt"), str(en_dir / "*.txt"),
                                        mixture_ratio_yoruba=0.5, seed=1)
