"""Command line: ``ragbench run bench.yaml`` prints a results table and can write JSON."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from ragbench.bench import run, to_markdown
from ragbench.data import DatasetError, load_dataset
from ragbench.retrievers import EmbeddingError


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping")
    for key in ("dataset", "chunkers", "retrievers"):
        if key not in config:
            raise ValueError(f"{path}: missing '{key}'")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_cmd = sub.add_parser("run", help="run a benchmark config")
    run_cmd.add_argument("config", type=Path)
    run_cmd.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args(argv)

    try:
        config = _load_config(args.config)
        dataset = load_dataset((args.config.parent / config["dataset"]).resolve())
        results = run(dataset, config["chunkers"], config["retrievers"], int(config.get("k", 5)))
    except (ValueError, DatasetError, EmbeddingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"dataset: {dataset.name} ({len(dataset.documents)} documents, "
        f"{len(dataset.queries)} queries)\n"
    )
    print(to_markdown(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([r.as_dict() for r in results], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
