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
 if (data.get("deployment_authorized") is not True or
     data.get("rights_approval_for_public_pages") is not True or
     data.get("deployment_ref") != "refs/heads/master"):
  raise ValueError("Owner approved master-only site deployment required")
 auth=json.loads((ROOT/"docs/STAGE22_PUBLICATION_AUTHORIZATION_v1.json").read_text(encoding="utf-8"))
 if (auth.get("owner_authorized_in_conversation") is not True or
     auth.get("approved_for_website_publication") is not True or
     auth.get("repository") != "Mihails-Birjukovs/Customized_WhisperCPP" or
     auth.get("deployment_ref") != "refs/heads/master" or
     auth.get("source_files_count") != len(allowed) or
     data.get("publication_authorization_record") != "docs/STAGE22_PUBLICATION_AUTHORIZATION_v1.json"):
  raise ValueError("Approved publication record or curated source mismatch")
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
  if path.is_symlink() or path.suffix.lower() not in (".md",".css",".js"):
   raise ValueError("Unsafe site asset")
  if path.suffix.lower()==".js" and rel!="javascripts/mathjax.js":
   raise ValueError("Only reviewed numbered-MathJax script is permitted")
  content=path.read_bytes()
  if hashlib.sha256(content).hexdigest()!=data["files_sha256"][rel]:
   raise ValueError("Document content SHA changed: "+rel)
  text=content.decode("utf-8",errors="strict")
  if any(x.search(text) for x in BLOCKED):
   raise ValueError("Private or credential-like content: "+rel)
  if re.search(r"(?i)(?:data:|src\s*=\s*['\"]https?://)",text):
   raise ValueError("Unreviewed embedded external asset: "+rel)
 conf=(ROOT/"zensical.toml").read_text(encoding="utf-8")
 script=(SITE/"javascripts/mathjax.js").read_text(encoding="utf-8")
 if not all(x in conf for x in ("pymdownx.arithmatex.generic = true","mathjax@3.2.2","javascripts/mathjax.js")) or 'tags: "all"' not in script or "document$.subscribe" not in script:
  raise ValueError("Missing numbered MathJax configuration")
 if "index.md" not in allowed:raise ValueError("Missing home page")
 return {"site_files":len(actual),"verified":True,"deployment":"approved_master_only"}
if __name__=="__main__":
 print(json.dumps(audited(),indent=2))
