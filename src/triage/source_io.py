"""Source IO and integrity checking module.
Handles JSONL reading/writing, cryptographic hashing, and split validation.
"""

from typing import Dict, List, Any, Optional
import json
import hashlib
import os


def compute_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str, expected_split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load JSONL file with strict line number error reporting and validation."""
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                data = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupt JSONL at {path}:{idx}: {e}") from e

            if not isinstance(data, dict) or "id" not in data or "messages" not in data:
                raise ValueError(f"Malformed schema at {path}:{idx}: expected 'id' and 'messages' keys")

            rows.append(data)

    if expected_split is not None:
        validate_split_roles(rows, expected_split, path)

    return rows


def validate_split_roles(rows: List[Dict[str, Any]], split: str, path: str = ""):
    """Validate that message roles match the expected split contract."""
    for idx, r in enumerate(rows, start=1):
        msgs = r.get("messages", [])
        if split in {"train", "dev"}:
            if len(msgs) != 3:
                raise ValueError(f"Row {r.get('id', idx)} in {split} must have 3 messages (system, user, assistant), got {len(msgs)}")
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"Row {r.get('id', idx)} in {split} invalid roles: {roles}")
        elif split == "test":
            if len(msgs) != 2:
                raise ValueError(f"Row {r.get('id', idx)} in test must have 2 messages (system, user), got {len(msgs)}")
            roles = [m.get("role") for m in msgs]
            if roles != ["system", "user"]:
                raise ValueError(f"Row {r.get('id', idx)} in test invalid roles: {roles}")


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    """Write rows atomically to a JSONL file."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def get_source_manifest(paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Compute manifest (sha256, bytes, row count) for a dictionary of split paths."""
    manifest = {}
    for name, path in paths.items():
        if os.path.exists(path):
            sha = compute_sha256(path)
            size = os.path.getsize(path)
            rows = load_jsonl(path)
            manifest[name] = {
                "path": path,
                "sha256": sha,
                "bytes": size,
                "rows": len(rows),
                "unique_ids": len(set(r["id"] for r in rows)),
            }
        else:
            manifest[name] = {"path": path, "status": "MISSING"}
    return manifest


def scan_cross_split_overlap(train_rows: List[Dict[str, Any]], dev_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Diagnostic tool: detect tickets that appear verbatim in both train and dev.
    Used for audit transparency, NEVER to leak dev labels into training.
    """
    dev_map = {r["messages"][1]["content"].strip(): r for r in dev_rows}
    overlap = []
    label_matches = 0

    for tr in train_rows:
        u = tr["messages"][1]["content"].strip()
        if u in dev_map:
            dr = dev_map[u]
            tr_label = tr["messages"][-1]["content"].strip()
            dr_label = dr["messages"][-1]["content"].strip()
            same_label = (tr_label == dr_label)
            if same_label:
                label_matches += 1
            overlap.append({
                "train_id": tr["id"],
                "dev_id": dr["id"],
                "identical_label": same_label,
            })

    return {
        "overlap_count": len(overlap),
        "exact_label_matches": label_matches,
        "overlap_details": overlap,
    }
