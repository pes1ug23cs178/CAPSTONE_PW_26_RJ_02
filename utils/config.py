"""Config loading: a run YAML merged with configs/paths.yaml, plus CLI overrides.

Usage in a script:
    cfg = load_config("configs/b1_swin.yaml", seed=42, overrides=["optim.epochs=1"])
    cfg["paths"]["data_root"]  -> absolute pathlib.Path
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# GRACE_REPO_ROOT is set by scripts/run_snapshot.py when a job runs from a frozen copy of the code.
REPO_ROOT = Path(os.environ.get("GRACE_REPO_ROOT") or Path(__file__).resolve().parents[1]).resolve()


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_paths(paths_file: Optional[str] = None) -> Dict[str, Path]:
    paths_file = paths_file or str(REPO_ROOT / "configs" / "paths.yaml")
    with open(paths_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {k: _resolve(v) for k, v in raw.items()}


def _parse_value(text: str) -> Any:
    # yaml parses numbers, bools, lists, null the same way the config files do
    return yaml.safe_load(text)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return cfg


def load_config(path: str, seed: Optional[int] = None, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p  # configs always come from the real repository (they may be written mid-queue)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)
    cfg = apply_overrides(cfg, overrides or [])
    if seed is not None:
        cfg["seed"] = int(seed)
    cfg.setdefault("seed", 42)
    cfg["config_path"] = str(Path(path).as_posix())
    cfg["paths"] = {k: str(v) for k, v in load_paths().items()}
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--override", nargs="*", default=[], help="key.sub=value pairs")
    return parser
