from pathlib import Path
from typing import Dict, Any


def get_directory(child_dir: str) -> Path:
    path = Path.cwd() / child_dir
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder already exists: {path}")
    else:
        print(f"Folder was created: {path}")

    return path


def pretty_print(d: Dict[str, Any], indent=0) -> None:
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pretty_print(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))
