#!/usr/bin/env python
"""Demo-only helper: download a Kaggle notebook's output and extract the
trained adapter straight into models/single_run/, automating everything
EXCEPT the one-time Kaggle login Kaggle itself requires (no code can skip
that -- it is how Kaggle verifies you own the notebook).

One-time setup (only needed once, ever):
    Run:  .venv\\Scripts\\kaggle.exe auth login   (opens your browser; approve once)
    (Alternatively set the KAGGLE_API_TOKEN environment variable.)

Usage:
    python -X utf8 scripts/fetch_kaggle_output.py <owner>/<notebook-slug>
(-X utf8 is needed on Windows: the notebook log has characters the default
Windows text encoding cannot write.)
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="Kaggle kernel slug, e.g. ayoolakayodebabak/notebook434cb61681")
    parser.add_argument("--download-dir", default="kaggle_download",
                         help="scratch folder for the raw download (default: kaggle_download)")
    parser.add_argument("--target", default="models/single_run",
                         help="where to extract the trained adapter (default: models/single_run)")
    args = parser.parse_args()

    # The kaggle package can authenticate several ways (OAuth login, a
    # KAGGLE_API_TOKEN environment variable, ~/.kaggle/access_token, or a
    # legacy kaggle.json), so rather than guessing which file exists, just
    # try to log in and explain the simplest fix if it fails. Importing the
    # package can itself exit the process on failure, hence SystemExit.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except (Exception, SystemExit):
        print("\nERROR: could not log in to Kaggle.\n\n"
              "This is the one-time Kaggle login step -- no script can skip it. Simplest fix:\n"
              "  .venv\\Scripts\\kaggle.exe auth login\n"
              "(opens your browser; approve it once, nothing to copy or paste). Then re-run this script.")
        sys.exit(1)

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading output of kernel '{args.kernel}' into {download_dir}/ ...")
    api.kernels_output(args.kernel, path=str(download_dir))

    zips = list(download_dir.glob("*.zip"))
    if not zips:
        print(f"ERROR: no .zip file found among the downloaded files: "
              f"{[p.name for p in download_dir.iterdir()]}")
        sys.exit(1)
    if len(zips) > 1:
        print(f"WARNING: more than one zip found ({[p.name for p in zips]}), using the first one.")
    adapter_zip = zips[0]
    print(f"Found {adapter_zip}")

    target = Path(args.target)
    if target.exists():
        backup = target.parent / f"{target.name}_backup"
        i = 1
        while backup.exists():
            backup = target.parent / f"{target.name}_backup_{i}"
            i += 1
        print(f"Existing {target} found -- moving it to {backup} instead of overwriting.")
        shutil.move(str(target), str(backup))

    target.mkdir(parents=True)
    with zipfile.ZipFile(adapter_zip) as zf:
        zf.extractall(target)

    # A zip sometimes contains one extra nested folder (seen before with
    # this exact notebook) -- if adapter_config.json isn't directly inside
    # target, but is one level deeper, flatten it automatically.
    if not (target / "adapter_config.json").exists():
        nested = list(target.glob("*/adapter_config.json"))
        if nested:
            nested_dir = nested[0].parent
            print(f"Zip had an extra nested folder ({nested_dir.name}) -- flattening it.")
            for item in nested_dir.iterdir():
                shutil.move(str(item), str(target / item.name))
            nested_dir.rmdir()

    if (target / "adapter_config.json").exists():
        print(f"\nSUCCESS: adapter extracted to {target}/")
        print((target / "adapter_config.json").read_text(encoding="utf-8")[:300])
    else:
        print(f"\nWARNING: extracted to {target}/ but no adapter_config.json found directly inside it "
              f"or one level deeper -- check its contents manually:")
        for p in target.rglob("*"):
            print(" ", p)


if __name__ == "__main__":
    main()
