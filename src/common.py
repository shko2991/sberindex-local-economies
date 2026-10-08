"""Общие утилиты: конфиг, пути, логирование, UTF-8 вывод на Windows."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def load_config(p: str | Path | None = None) -> dict:
    p = Path(p or os.environ.get("SBER_CONFIG") or ROOT / "config" / "config.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg: dict, key: str, *parts: str) -> Path:
    """Путь из раздела paths конфига (+ подпути). Абсолютные пути не трогаются."""
    base = Path(cfg["paths"][key])
    p = base if base.is_absolute() else ROOT / base
    p = p.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


def rng(cfg: dict, offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(cfg.get("seed", 42)) + offset)
