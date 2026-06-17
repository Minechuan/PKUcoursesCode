from __future__ import annotations

import re

TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+|"
    r"FY\d{2,4}|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"\d+(?:\.\d+)?%?|"
    r"\$+",
    re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]
