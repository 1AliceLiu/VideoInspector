# Camera 2676 candidate observation

Image: safety-detector:video-integrity-candidate. Physical GPU: 1.
Independent Docker container: video-observe-2676, restart unless stopped.
No webhook sender or VLM calls exist in this observation pipeline.
The production detector is untouched. Full existing small-model inference chain
runs before sampling; annotated images show person/head/helmet detections and
vest classifier labels, not platform alarm decisions.

Every 10 seconds: raw_*.jpg (original), annotated_*.jpg (same-frame boxes),
meta_*.json (frame/PTS/detections, time in Beijing). samples.csv is an append-only
index; status.json records the latest sample. observer.log and Docker logs show
errors. Automatic retention deletion is disabled. All generated images and metadata are retained.
The green-pixel metric is only a simple diagnostic; green objects are normal
and a low value does not certify that all video artifacts are absent.

Check status:
    docker ps --filter name=video-observe-2676
Follow logs:
    docker logs --tail 50 -f video-observe-2676
Restart observation only:
    docker restart video-observe-2676
Stop observation only:
    docker stop video-observe-2676

All files here are generated from camera 2676 by the observer, with no alarm
pushes. Stopping this observer does not stop the production detector.

Owner policy: read/write/add allowed; deleting files, directories or data is prohibited.
Candidate helmet weights are pinned to the November 5 checkpoint even if production config changes.
