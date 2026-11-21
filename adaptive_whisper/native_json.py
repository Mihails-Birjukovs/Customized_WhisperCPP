"""Lossless, separate derived JSON for native token-boundary UTF-8 fragments.

The source native -ojf file is immutable. Invalid token text bytes become
explicit hex fields, never U+FFFD replacement characters. Derived JSON is NOT
the original and must retain its provenance.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


class NativeJsonIntegrityError(ValueError):
    pass


def _raw(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogateescape")


def _valid(text: str) -> bool:
    try:
        _raw(text).decode("utf-8", errors="strict")
        return True
    except UnicodeDecodeError:
        return False


def _has_surrogate(value: Any) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(ch) <= 0xDFFF for ch in value)
    if isinstance(value, list):
        return any(_has_surrogate(x) for x in value)
    if isinstance(value, dict):
        return any(_has_surrogate(k) or _has_surrogate(v) for k, v in value.items())
    return False


def prepare_lossless_canonical(native_bytes: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Requires valid segment text and strict UTF-8 after joining token bytes.

    Returns serializable canonical payload and a text-free audit summary.
    Refuses other malformed JSON or non-token UTF-8 damage.
    """
    sha = hashlib.sha256(native_bytes).hexdigest()
    try:
        decoded = native_bytes.decode("utf-8-sig", errors="surrogateescape")
        payload = json.loads(decoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NativeJsonIntegrityError("Cannot parse reversible native JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("transcription"), list):
        raise NativeJsonIntegrityError("Native schema lacks transcription list")
    derived = copy.deepcopy(payload)
    converted = 0
    affected = 0
    segments = derived["transcription"]
    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict) or not isinstance(seg.get("text"), str):
            raise NativeJsonIntegrityError(f"Invalid segment text schema at {idx}")
        if not _valid(seg["text"]):
            raise NativeJsonIntegrityError(f"Segment text itself malformed at {idx}")
        tokens = seg.get("tokens")
        if not isinstance(tokens, list):
            raise NativeJsonIntegrityError(f"No token list at segment {idx}")
        token_bytes = []
        changed = False
        for tok in tokens:
            if not isinstance(tok, dict) or not isinstance(tok.get("text"), str):
                raise NativeJsonIntegrityError(f"Invalid token text schema at {idx}")
            b = _raw(tok["text"])
            token_bytes.append(b)
            if not _valid(tok["text"]):
                # Explicit binary fragment, not a fabricated text codepoint.
                tok["text"] = None
                tok["text_raw_bytes_hex"] = b.hex()
                tok["text_encoding"] = "utf8_fragment_not_independently_decodable"
                converted += 1
                changed = True
        try:
            b"".join(token_bytes).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise NativeJsonIntegrityError(
                f"Concatenated token bytes still invalid at segment {idx}"
            ) from exc
        affected += int(changed)
    # Any other undecodable bytes outside token text are a hard failure.
    if _has_surrogate(derived):
        raise NativeJsonIntegrityError("Undecodable bytes remain outside token text")
    provenance = {
        "schema": "whispercpp/lossless-derived-token-json/v1",
        "source_sha256": sha,
        "original_native_preserved": True,
        "native_strict_utf8": _valid(decoded),
        "converted_token_fields": converted,
        "affected_segments": affected,
        "transcription_segments": len(segments),
        "method": "surrogateescape reversible parse; undecodable token bytes stored as exact hex",
        "accuracy_claim": "none; encoding repair only, no acoustic verification",
    }
    return derived, provenance


def write_canonical_separate(native_path: Path, dest_path: Path,
                             audit_path: Path) -> dict[str, Any]:
    if dest_path.exists() or audit_path.exists():
        raise FileExistsError("Refusing overwrite of canonical artifact or audit")
    source_bytes = native_path.read_bytes()
    payload, provenance = prepare_lossless_canonical(source_bytes)
    # Include a dedicated provenance namespace in the new file.
    if "_whispercpp_derived_provenance" in payload:
        raise NativeJsonIntegrityError("Unexpected pre-existing provenance key")
    payload["_whispercpp_derived_provenance"] = provenance
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    data = rendered.encode("utf-8", errors="strict") + b"\n"
    # Verify reversible reconstruction of the *entire* native data structure,
    # including raw bytes of each token, not just its derived hex field.
    parsed = json.loads(data.decode("utf-8", errors="strict"))
    original = json.loads(source_bytes.decode("utf-8-sig", errors="surrogateescape"))
    reconstructed = copy.deepcopy(parsed)
    reconstructed.pop("_whispercpp_derived_provenance")
    for segment in reconstructed["transcription"]:
        for token in segment["tokens"]:
            if token.get("text_encoding") == "utf8_fragment_not_independently_decodable":
                fragment = bytes.fromhex(token.pop("text_raw_bytes_hex"))
                token.pop("text_encoding")
                token["text"] = fragment.decode("utf-8", errors="surrogateescape")
    if reconstructed != original:
        raise NativeJsonIntegrityError("Canonical JSON failed original-structure round trip")
    provenance["structural_and_raw_token_roundtrip_verified"] = True
    provenance["canonical_sha256"] = hashlib.sha256(data).hexdigest()
    # Keep source SHA recorded in both output and audit.
    dest_path.write_bytes(data)
    audit_path.write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    return provenance
