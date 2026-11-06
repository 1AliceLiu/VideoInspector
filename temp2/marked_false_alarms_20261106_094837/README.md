# Marked false-alarm review

Recent scope: 26 no_helmet cases, including all 13 additions captured during this review.
68 total marked records and original images are archived; 42 older records are
from other algorithms and are not attributed to the November helmet weights.

## Confirmed findings

1. New weights confuse visible orange hard hats with head. At equal 960 input:

| ID | Camera | Old weights | New weights |
| --- | --- | --- | --- |
| 26329 | 1912 | helmet 0.890 | head 0.700 |
| 26328 | 1912 | helmet 0.864 | head 0.811 |
| 26323 | 1912 | helmet 0.887 | head 0.827 |

The 640/960 control is in controlled_size_comparison.json. Old weights remain
correct on these three at 960; the new weights are also sensitive to input size.
This supports a weight-related regression on these examples, not just resolution.

2. ID 26320: the raised circular tool is detected as head (0.720). The current
whole-person center-in-box association lets this non-head object satisfy the
no_helmet rule. replay_26320.jpg shows the implicated box.

3. Several samples contain severe block corruption, ghosting or missing head
pixels. Candidate video recovery is running only in the 2676 observer, not on
the production alarm pipeline, so production examples can still be corrupted.

4. Some marked examples show brimmed hats/head coverings. Recheck the human
label before using them for training: fabric hats are not compliant hard hats.

## Timing and confidence limits

All 26 records matched evidence timing logs: 80-400 ms skew, median 100.5 ms.
This is a real mismatch, but it cannot explain all stationary-hard-hat errors.
Saved JPEGs are annotated snapshots rather than exact decision frames. Replay
is FP16 PyTorch, not exact DeepStream/TensorRT preprocessing. These cases alone
do not measure precision/recall or establish whether model rollback is safe.

## Artifacts

originals/: untouched downloaded evidence. marked_records.json: platform labels.
replay_*.jpg: magenta head / cyan helmet reference replay boxes on evidence.
replay_results.json, old_new_comparison.json, controlled_size_comparison.json.
evidence_log_matches.json: matched camera/time records, in Beijing time.
analysis.json: structured findings. No platform flags or production settings changed.

Next: validate candidate decoding on clean frames, collect a balanced set of real
violations and hard-hat negatives, then test weight correction and association
changes together for false positives AND missed violations. Do not infer safety
from false-positive-only replay or raise thresholds without recall checks.

## Parent-checkpoint control and final additions

The October 29 parent is not uniformly correct. At 960 it reports:
26329 helmet .737 (child head .700); 26328 helmet .515 / head .442
(child head .811); 26323 already head .742 (child head .827).
The circular tool in 26320 is misdetected as helmet .649 by the parent and head
.720 by the child. Some weaknesses therefore predate the final fine-tune;
others worsen after it. The earliest production August weights recognize the
three orange-hard-hat cases as helmet .890/.864/.887 at the same 960 input.

26410: visible white hard hat, original confidence .860; saved-image replay
cannot reproduce the head detection. Cause remains unresolved without the
original decision pixels. 26406: strong color/ghosting damage. 26396: apparent
red fabric cap; the human false-alarm label needs review under hard-hat-only rules.

weight_evidence.jpg visualizes equal-size old/new comparisons. Parent results
are in parent_checkpoint_comparison.json. Model weights, thresholds, platform
labels, and production configuration were not changed. The 2676 observer remains
running on physical GPU 1; these offline replays used otherwise idle GPU 2.

## Runtime state verified at 2026-11-06 12:55 Beijing time

During this read-only analysis, an external action changed production config at
11:49 and restarted safety-detector at 11:50:45 Beijing time. Production now
points to the August 4 helmet checkpoint. This investigation did not perform
that production change. The marked examples reviewed here end at 10:01:50,
therefore describe the pre-change November model, not the now-running weights.
The 2676 candidate observer still uses the November 5 checkpoint, pinned in its
script, so later production/candidate comparisons must distinguish the weights.
The replay was repeated with an explicit November checkpoint path after noticing
the config change; replay_results.json records the checkpoint and input settings.

Owner permissions updated: reading, writing and adding are allowed; deleting
server files/directories/data is forbidden. Automatic image retention deletion
was disabled in the 2676 observer; no production service was restarted for this.
