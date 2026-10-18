"""Flag f-strings that reuse their own quote character inside {...}. That only works on Python 3.12+."""
import io, sys, tokenize
from pathlib import Path

bad = 0
for path in sorted(Path(__file__).resolve().parent.parent.rglob("*.py")):
    if any(part in {".venv", "venv", "data", "data_synthetic", "__pycache__"} for part in path.parts):
        continue
    src = path.read_text()
    stack = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception as e:
        print(f"{path}: could not tokenize ({e})"); bad += 1; continue
    for tok in toks:
        name = tokenize.tok_name.get(tok.type, "")
        if name == "FSTRING_START":
            stack.append((tok.string[-1], tok.start))
        elif name == "FSTRING_END":
            stack.pop()
        elif name == "STRING" and stack:
            q = stack[-1][0]
            body = tok.string.lstrip("rbfRBF")
            if body and body[0] == q:
                print(f"{path}:{tok.start[0]}: string using {q} inside an f-string delimited by {q} (needs Python 3.12)")
                bad += 1
print("OK: no 3.12-only f-strings found" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
