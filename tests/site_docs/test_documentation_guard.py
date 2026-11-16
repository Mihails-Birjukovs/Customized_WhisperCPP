import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from validate_stage22_docs import audited,MANIFEST,SITE
class Stage22DocsGuardTests(unittest.TestCase):
 def test_actual_source_allowlist(self):
  self.assertTrue(audited()["verified"])
 def _fixture(self,mutate):
  with tempfile.TemporaryDirectory() as temp:
   base=Path(temp);docs=base/"docs";docs.mkdir()
   current=json.loads(MANIFEST.read_text(encoding="utf-8"))
   import shutil
   shutil.copytree(SITE,docs/"site")
   target=docs/"manifest.json"
   mutate(current,docs/"site")
   target.write_text(json.dumps(current),encoding="utf-8")
   with self.assertRaises(ValueError):
    audited(docs/"site",target)
 def test_footer_contrast_has_explicit_high_specificity_rules(self):
  css=(SITE/"stylesheets/inm-documentation-v1.css").read_text(encoding="utf-8")
  self.assertIn("footer.md-footer .md-footer-meta .md-copyright",css)
  self.assertIn("color: #ffffff;",css)
  self.assertIn("a:focus-visible",css)
 def test_unexpected_source_is_blocked(self):
  self._fixture(lambda data,root:(root/"secret.wav").write_bytes(b"abc"))
 def test_modified_content_hash_is_blocked(self):
  self._fixture(lambda data,root:(root/"index.md").write_text("changed",encoding="utf-8"))
 def test_no_public_deployment_without_distinct_approval(self):
  self._fixture(lambda data,root:data.update(deployment_authorized=False))
 def test_missing_rights_approval_rejected(self):
  self._fixture(lambda data,root:data.update(rights_approval_for_public_pages=False))
 def test_nonmaster_deployment_rejected(self):
  self._fixture(lambda data,root:data.update(deployment_ref="refs/heads/other"))
 def test_unsafe_embedded_file_path_rejected(self):
  def change(data,root):
   p=root/"index.md"
   p.write_text(p.read_text(encoding="utf-8")+"\nX:\\\\Private\\\\audio.wav",encoding="utf-8")
   import hashlib
   data["files_sha256"]["index.md"]=hashlib.sha256(p.read_bytes()).hexdigest()
  self._fixture(change)
if __name__=="__main__":unittest.main()
