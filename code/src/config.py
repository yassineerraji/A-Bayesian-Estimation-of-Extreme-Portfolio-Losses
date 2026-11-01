"""Load the project settings from config.toml."""

import tomllib
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]  # the code/ folder, which holds config.toml


def load_config():
    """Read config.toml and return it as a dict, with every [paths] entry as an absolute Path."""
    with open(CODE_DIR / "config.toml", "rb") as f:
        cfg = tomllib.load(f)
    cfg["paths"] = {name: (CODE_DIR / path).resolve() for name, path in cfg["paths"].items()}
    return cfg
