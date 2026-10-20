"""Verify all release assets against SHA256SUMS before extraction."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parent
for line in (root/"SHA256SUMS").read_text().splitlines():
    expected,name=line.split(maxsplit=1)
    p=root/name.strip().lstrip("*")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    assert h.hexdigest()==expected, p.name
    print("PASS",p.name)
