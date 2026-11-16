# Stage 22 — documentation and repository hygiene review

**Scope:** documentation-only worktree based on 442aa2a01c1319041354aaabfe82b3dc2ccefb2a. No adaptive decoding defaults, original WAVs, previous experimental transcripts, or native binaries are changed.

Changes: preserve old upstream README at repository root; replace misleading fork front page; archive historic project scope; add curated site_docs, INM-inspired CSS, exact public source-file allowlist, fail-closed static validator and strict Zensical preview CI.

## Independent acceptance gates

- [x] Scientific claims checked against frozen private Stage 20 aggregate hash.
- [x] Scope separates structural repetition from acoustic WER/CER and full recording coverage.
- [x] No official logo or redistributed research audio included.
- [x] Local Zensical 0.0.67 strict source build and generated HTML/CSS/navigation checks passed on Windows Python 3.10 (2026-11-15).
- [ ] Reviewed draft PR by owner for wording, authorship, publication rights and software provenance.
- [ ] Pages eligibility and owner authorization for live GitHub Pages deployment.
- [ ] Live browser checks (stylesheets, links, mobile, logo rights, accessibility, actual deployed SHA).
- [ ] Separate experimental-code merge and human transcription validation gates.

**The source repository is public, so draft PR content is already visible.** Only curated, non-private documentation should be committed. The CI preview job uses contents:read and does not deploy.

Proposed page URL in configuration is not evidence that a Pages site exists. The upstream-readme archive and historic scope archive are retained for reproducibility and rollback. Branch cleanup requires its own audit.
