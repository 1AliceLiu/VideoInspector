# Video integrity fix (2026-11-06)

Scope: video corruption only. No model, alarm, or VLM review changes.

Reproduction: sandbox/video-distortion/ds_stress.py throttles one decoded camera
with a 200ms identity element and a small source queue. With nvurisrcbin's default
drop-on-latency=true, captured frame 90 is largely green/corrupted. With false,
the frame is clear. ds_bounded.py additionally drops stale complete RAW frames
before streammux; it stays clear under the same downstream slowdown.
These are short controlled samples, not proof that every real-world corruption
has this cause. TCP capture also saw initial HEVC reference-frame errors.

Implementation: source.preserve_bitstream defaults false for compatibility.
When true, disable RTP jitterbuffer overflow drops and insert one two-buffer
leaky-downstream RAW queue per source, after nvurisrcbin, before nvstreammux.
Never insert leaky queues in front of an inter-frame decoder.

Production config and container remain unchanged. Candidate configuration is
sandbox/video-distortion/candidate-config.yaml. Enable source.preserve_bitstream
only when deploying the tested image in an approved maintenance window.
Rollback: disable source.preserve_bitstream and restart the detector, or return
to the previous image. A restart causes a brief detection gap.

Previous same-frame evidence / shadow-review changes are archived in
sandbox/evidence-sync-proposal and are NOT part of this patch.

Artifacts: temp2/video-distortion/slow_1168_drop1_90.jpg (bad),
slow_1168_drop0_90.jpg (clear but buffered), bounded_1168_drop0_90.jpg (clear).

Startup recovery: joining an H.265 stream mid-GOP produced green startup frames
even with RTP preservation. An inherited nvurisrcbin source registers recursive
child-added callbacks. Parser output rejects delta/corrupt frames until a clean
keyframe, resetting on new segments/flushes; decoder flags corrupt output for
discard and requests sync points when supported. The protected-source startup
experiment produced 3/3 clear samples; discard-corrupted alone was insufficient.
This may delay initial image availability until the next camera keyframe.
The feature stays opt-in and the production container is untouched.

Final validation (2026-11-06, Beijing time): 34 targeted tests passed. The actual
PipelineBuilder candidate ran for 35 seconds under 200ms downstream delay,
rejected 116 pre-keyframe compressed buffers at startup (~6 seconds), and
produced 4 sampled frames (30/60/90/120) without large green areas.
The control frames 30/60/90 were 97.8-98.8% bright green by a diagnostic pixel
heuristic. This detects the reproduced green corruption, not all visual defects.
comparison.jpg shows a control/candidate pair.

Image: safety-detector:video-integrity-candidate (not latest).
The latest tag was restored to the still-running baseline to avoid accidental
deployment of the unrelated archived evidence-sync image.
Production deployment and full 44-camera validation have NOT been performed.

Prepared deployment override (NOT executed):
    docker compose -f deploy/docker-compose.yml -f sandbox/video-distortion/deploy-candidate.yml up -d --no-build detector
Run from /home/pzq/projects/VideoInspector only in a maintenance window.
Rollback to the existing baseline image/config:
    docker compose -f deploy/docker-compose.yml up -d --no-build detector
Candidate config/override are local artifacts, not committed; the candidate
config contains the same private stream endpoints as production and is mode 600.
Neither command was executed during this investigation.

Detection reliability: corruption occurs before inference in the reproduced
case, so small-model results on those damaged frames may be false positives
or false negatives. The source protection is placed before streammux/nvinfer.
It does not detect every possible visual artifact; a camera that encodes an
already damaged picture as a valid keyframe is outside this recovery mechanism.
