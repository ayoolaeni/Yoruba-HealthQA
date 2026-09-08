#!/usr/bin/env python
"""Phase 1.1 -- Ingest consumer health content from configs/data.yaml sources.

Refuses any source with licence: null (rule 1: never invent data / never use
content without a recorded licence). Writes data/raw/<source>/<slug>.json with
URL, retrieval date, licence, and SHA-256 of the raw text.

Usage:
    python scripts/01_ingest.py --config configs/data.yaml --out data/raw
    python scripts/01_ingest.py --config configs/data.yaml --out data/raw --sources who_factsheets medlineplus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest_sources import LicenceMissingError, build_document, check_licence_or_raise, write_document
from src.utils.logging import MissingInputError, get_logger
from src.utils.manifest import RunManifest
from src.utils.seeding import seed_everything

logger = get_logger("01_ingest")


def fetch_html_text(url: str, timeout: int = 30) -> str:
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "yoruba-healthqa-research/1.0"})
    resp.raise_for_status()
    # Use raw bytes (resp.content), NOT resp.text: when a server's
    # Content-Type header omits a charset (MedlinePlus does this),
    # `requests` falls back to the HTTP spec default of ISO-8859-1 even when
    # the actual bytes are UTF-8, silently mangling every curly
    # quote/apostrophe (verified: this happened on medlineplus.gov pages).
    # BeautifulSoup's own encoding detection (UnicodeDammit, from the bytes
    # and any <meta charset> tag) is far more reliable here.
    soup = BeautifulSoup(resp.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def ingest_web_source(name: str, cfg: dict, out_dir: Path, sleep_s: float = 1.0) -> list[Path]:
    check_licence_or_raise(name, cfg)
    written = []
    slugs = cfg.get("slugs") or []
    if not slugs:
        logger.warning(f"[{name}] no slugs configured yet -- nothing to fetch. "
                        f"Add URLs to configs/data.yaml sources.{name}.slugs.")
        return written
    for slug in slugs:
        url = cfg["base_url"].rstrip("/") + "/" + slug.lstrip("/")
        logger.info(f"[{name}] fetching {url}")
        try:
            text = fetch_html_text(url)
        except Exception as e:
            logger.error(f"[{name}] failed to fetch {url}: {e}")
            continue
        if not text.strip():
            logger.warning(f"[{name}] fetched empty text from {url}, skipping")
            continue
        doc = build_document(name, cfg, url=url, raw_text=text)
        safe_slug = slug.replace("/", "_")
        path = write_document(doc, out_dir, safe_slug)
        logger.info(f"[{name}] wrote {path}")
        written.append(path)
        time.sleep(sleep_s)  # be polite to the source server
    return written


def ingest_manual_source(name: str, cfg: dict, out_dir: Path) -> list[Path]:
    check_licence_or_raise(name, cfg)
    manual_dir = out_dir / name / "manual"
    if not manual_dir.exists():
        raise MissingInputError(
            str(manual_dir), stage="01_ingest",
            hint=f"'{name}' is kind: manual -- place cleared source text/PDF-extracted "
                 f"text files in {manual_dir} before running ingest for this source.",
        )
    written = []
    for src_file in sorted(manual_dir.glob("*.txt")):
        text = src_file.read_text(encoding="utf-8")
        doc = build_document(name, cfg, url=None, raw_text=text)
        path = write_document(doc, out_dir, src_file.stem)
        written.append(path)
    if not written:
        logger.warning(f"[{name}] manual source directory {manual_dir} has no .txt files yet")
    return written


def ingest_dataset_source(name: str, cfg: dict, out_dir: Path) -> list[Path]:
    check_licence_or_raise(name, cfg)
    from datasets import load_dataset

    logger.info(f"[{name}] loading HuggingFace dataset {cfg['base_url']}")
    ds = load_dataset(cfg["base_url"].rsplit("/", 2)[-2] + "/" + cfg["base_url"].rsplit("/", 1)[-1])
    written = []
    out_subdir = out_dir / name
    out_subdir.mkdir(parents=True, exist_ok=True)
    for split_name, split in ds.items():
        text_blob = "\n".join(str(row) for row in split)
        doc = build_document(name, cfg, url=cfg["base_url"], raw_text=text_blob)
        path = write_document(doc, out_dir, split_name)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--sources", nargs="*", default=None,
                         help="Subset of source names to ingest; default = all sources in config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise MissingInputError(str(config_path), stage="01_ingest")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = seed_everything(cfg.get("seed", 42))
    out_dir = Path(args.out)
    sources = cfg["sources"]
    names = args.sources or list(sources.keys())

    with RunManifest(stage="01_ingest", config={"config_path": str(config_path), "sources": names}, seed=seed) as run:
        refused = []
        total_written = []
        for name in names:
            if name not in sources:
                logger.error(f"unknown source '{name}' -- not in {config_path}")
                continue
            source_cfg = sources[name]
            kind = source_cfg.get("kind")
            try:
                if kind == "web":
                    written = ingest_web_source(name, source_cfg, out_dir)
                elif kind == "manual":
                    written = ingest_manual_source(name, source_cfg, out_dir)
                elif kind == "dataset":
                    written = ingest_dataset_source(name, source_cfg, out_dir)
                else:
                    logger.error(f"[{name}] unknown source kind '{kind}'")
                    continue
            except LicenceMissingError as e:
                logger.error(str(e))
                refused.append(name)
                continue
            total_written.extend(written)

        for p in total_written:
            run.record_output(str(p))

        logger.info(f"ingested {len(total_written)} documents across {len(names) - len(refused)} sources")
        if refused:
            logger.warning(f"refused (no licence recorded): {refused}")


if __name__ == "__main__":
    main()
