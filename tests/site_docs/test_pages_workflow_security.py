"""Fail-closed static safety contracts for Stage 22 Pages publishing."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/".github/workflows/stage22-pages.yml"
class PagesWorkflowSafetyTests(unittest.TestCase):
 def setUp(self):
  self.text=SOURCE.read_text(encoding="utf-8")
  self.assertIn("\n  deploy:\n",self.text)
  self.build,self.deploy=self.text.split("\n  deploy:\n",1)
 def test_only_master_push_and_manual(self):
  self.assertIn("on:\n  push:\n    branches:\n      - master",self.text)
  self.assertIn("  workflow_dispatch:",self.text)
  self.assertNotIn("pull_request:",self.text)
  self.assertNotIn("pull_request_target:",self.text)
 def test_both_jobs_enforce_master_and_repository(self):
  self.assertEqual(self.text.count("github.ref == 'refs/heads/master'"),2)
  self.assertEqual(self.text.count("github.repository == 'Mihails-Birjukovs/Customized_WhisperCPP'"),2)
 def test_deployment_credentials_only_in_deploy_job(self):
  self.assertIn("permissions: {}",self.build)
  self.assertIn("contents: read",self.build)
  self.assertNotIn("pages: write",self.build)
  self.assertNotIn("id-token: write",self.build)
  self.assertEqual(self.deploy.count("pages: write"),1)
  self.assertEqual(self.deploy.count("id-token: write"),1)
 def test_deploy_requires_validated_build_and_pages_environment(self):
  self.assertIn("needs: build",self.deploy)
  self.assertIn("name: github-pages",self.deploy)
  self.assertIn("actions/deploy-pages@v4",self.deploy)
  self.assertIn("actions/upload-pages-artifact@v4",self.build)
  self.assertIn("path: site",self.build)
 def test_strict_build_and_allowlist_required(self):
  for term in ("python scripts/validate_stage22_docs.py",
               "python -m unittest discover -s tests/site_docs -v",
               "zensical build --clean --strict",
               "python scripts/check_stage22_built_site.py",
               "zensical==0.0.67",
               "persist-credentials: false","lfs: false"):
   self.assertIn(term,self.build)
if __name__=="__main__":unittest.main()
