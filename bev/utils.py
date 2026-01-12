from __future__ import annotations
import os

def assert_file_exists(path: str, what: str = "file") -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {what}: {path}")

def assert_dir_exists(path: str, what: str = "directory") -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Missing {what}: {path}")

def info(msg: str) -> None:
    print(msg, flush=True)
