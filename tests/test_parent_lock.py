"""The five reused parent files must stay byte-identical to the export recorded in PARENT.lock."""
import hashlib
import json

from fusion_miniapp import PARENT_DIR


def test_parent_files_match_lock():
    lock = json.loads((PARENT_DIR / "PARENT.lock").read_text())
    assert lock["commit"] == "9768bcb"
    assert len(lock["sha256"]) == 5
    for rel, want in lock["sha256"].items():
        got = hashlib.sha256((PARENT_DIR / rel).read_bytes()).hexdigest()
        assert got == want, f"parent/{rel} differs from the locked copy"
