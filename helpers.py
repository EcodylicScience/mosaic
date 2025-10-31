from urllib.parse import quote, unquote

def to_safe_name(s: str) -> str:
    # encode EVERYTHING that could be problematic on any OS
    return quote(s, safe="")   # e.g. "task1/train/m1" -> "task1%2Ftrain%2Fm1"

def from_safe_name(safe: str) -> str:
    return unquote(safe)