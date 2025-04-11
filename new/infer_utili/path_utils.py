import sys

def add_to_syspath(path: str):
    if path not in sys.path:
        sys.path.append(path)