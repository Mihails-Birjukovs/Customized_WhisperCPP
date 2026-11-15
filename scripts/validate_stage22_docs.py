"""Stage 22 exact allowlist for review-only website source, not legal clearance."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/"site_docs"
MANIFEST=ROOT/"docs/STAGE22_SITE_SOURCE_ALLOWLIST_v1.json"
BLOCKED=[
 re.compile(r"(?i)(?:X|G):[\\/]"),
 re.compile(r"(?i)file:///"),
 re.compile(r"(?i)\\AI-Lab-Test\\"),
 re.compile(r"(?i)\b(?:secret|api[_-]?key|password)\s*[:=]\s*\S+"),
]
def audited(root=SITE,manifest=MANIFEST):
 data=json.loads(Path(manifest).read_text(encoding="utf-8"))
 allowed=data["approved_for_build_only"]
 if (data.get("deployment_authorized") is not False or
     data.get("rights_approval_for_public_pages") is not False):
  raise ValueError("No automatic approval for public deployment")
 if len(allowed)!=len(set(allowed)):
  raise ValueError("Duplicate document")
 actual=sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())
 if sorted(allowed)!=actual:
  raise ValueError("Unexpected/missing site source: extra="+str(sorted(set(actual)-set(allowed)))+" missing="+str(sorted(set(allowed)-set(actual))))
 if sorted(data["files_sha256"])!=actual:
  raise ValueError("Publication hash key mismatch")
 for rel in allowed:
  if rel.startswith("/") or ".." in Path(rel).parts:
   raise ValueError("Unsafe relative path")
  path=Path(root)/rel
  if path.is_symlink() or path.suffix.lower() not in (".md",".css"):
   raise ValueError("Unsafe site asset")
  content=path.read_bytes()
  if hashlib.sha256(content).hexdigest()!=data["files_sha256"][rel]:
   raise ValueError("Document content SHA changed: "+rel)
  text=content.decode("utf-8",errors="strict")
  if any(x.search(text) for x in BLOCKED):
   raise ValueError("Private or credential-like content: "+rel)
  if re.search(r"(?i)(?:data:|src\s*=\s*['\"]https?://)",text):
   raise ValueError("Unreviewed embedded external asset: "+rel)
 if "index.md" not in allowed:raise ValueError("Missing home page")
 return {"site_files":len(actual),"verified":True,"deployment":False}
if __name__=="__main__":
 print(json.dumps(audited(),indent=2))
