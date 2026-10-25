# Phase 0 transcription regression registry

The machine-readable inventory is `phase0_corpus_manifest.json`. Use the `folder_keys` mappings (relative to the **Custom WhisperCPP Dev** shared Drive) plus each record's unique filename glob. Each selector must resolve to exactly one file in its assigned folder.

- `suspected_output_repetition`: consecutive identical-text SRT segments with advancing timestamps. **Audio verification and human-corrected reference required** before treating as proven ASR errors.
- `candidate_negative_control`: chosen for lack of conspicuous repeated output, not demonstrated transcript accuracy.
- `untriaged`: paired source/output found but not classified.
- Every `legacy_txt`/`legacy_srt` is a historical transcription output, **not ground truth**. Production binary and model provenance remain unverified.
- Only metadata and locator patterns are tracked here; do not commit research audio, TXT/SRT transcript text, individual participants' filenames, authenticated Drive URLs/IDs, binaries or model weights.

Known exceptions: an additional same-name, same-size Lecture 3 WAV appears in the team-meeting audio folder (hash not yet compared); the 2026-03-10 TXT triggers a UTF-8 decoding error in the Drive text reader; `v1_for_redundancy.7z` and the historical Whisper binary archives have not had their contents audited.

Run checks with `python -m unittest discover -s tests/phase0 -v`.
