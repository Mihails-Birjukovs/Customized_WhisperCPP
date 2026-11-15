# Adaptive recovery

## Detection

The controller screens machine subtitle repetitions and duration anomalies, prioritizing suspicious intervals while retaining native output and source coordinates. Repeated text is a *machine pathology*, not proof of a repeated human utterance.

## Bounded correction attempts

For each selected source interval, the controller constructs short context windows or longer resets with fixed job/time budgets. It maps every cue back to the original audio clock and records caption-ownership or stitching conflicts.

A result may have zero repetition triggers but still require **join review**. Such a result must not be treated as an accepted final transcript.

## Uncaptioned audio screening

The experimental Stage 18 algorithm compares multithreshold RMS audio activity with machine subtitle coverage. Audio energy alone cannot distinguish speech from noise. It records candidate gaps, plans two distinct ASR contexts, and refuses machine text crossing unresolved boundaries; proposed words remain private review hypotheses.

**Stage 20:** one selected problem excerpt per 27 flagged recordings was reprocessed with a frozen retry policy. All 27 resulting previews were structurally free of detected loops. This does not prove that any given word is correct or that the remaining 66 historical trigger regions have all been processed.

The detailed chronological development ledger remains in docs/ADAPTIVE_FRAMEWORK_PROTOTYPE.md in the repository.
