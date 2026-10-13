# Automatic multi-view head reconstruction

The original distance application, its saved 30/60 cm calibration, keyboard
controls, and tests are preserved. This extension adds a separate head workflow
with local OpenCV, MediaPipe, COLMAP, NumPy, and Open3D. All processing and Python
dependencies run inside Docker. There are no accounts, memberships, paid APIs,
cloud inference, or runtime model downloads.

## What works, and what remains unresolved

| Component | Current behavior |
| --- | --- |
| Existing distance mode | Preserved; default `docker compose up` behavior |
| Intrinsic calibration | Automatic ChArUco corners; diverse-view selection; reprojection checks; stored K, five distortion coefficients, and resolution |
| Multi-view capture | Automatic head masks and overlapping frame selection; sharpness, exposure, texture, redundancy, and geometric-overlap checks |
| Fixed-webcam head turns | Explicit experimental mode; frontal anchor, both-side yaw coverage and feature-consistency gates; head-relative poses and arbitrary units |
| Reconstruction | Real COLMAP CPU SIFT, matching/geometric verification, and SfM; fixed calibrated intrinsics; subject-only masks |
| Dense reconstruction | Local CPU rectified stereo with left/right consistency, texture checks, mask support, and sparse-geometry support; sparse fallback clearly reported |
| Surface | Open3D outlier filtering and ball pivoting over observed points; gap bridges and tiny fragments removed; no hole filling or invented skull shape |
| 3D preview | CPU mesh rendering in the right pane; webcam on the left; arrows rotate the preview |
| Metric scale | Static ChArUco board, independently measured square size, registered multi-view corners, and rigid-fit/uncertainty checks |
| Facial anatomy | Automatic estimated glabella region, eyebrow peaks, chin/nose orientation and symmetry references from MediaPipe; no claim of precise anatomical point accuracy |
| Frontal hairline | Estimated local hair/face-skin segmentation boundary; associated with the same visible mesh vertices in multiple registered views |
| Ear anatomy | No suitable validated pretrained anatomical model is bundled. Default output is `UNRESOLVED`. A local four-keypoint YOLO ONNX interface is provided |
| Headband poles | Automatic-detector contract provided; no validated pole-position detector is bundled. `UNRESOLVED` |
| Kang V/P/FL/FR | Gated geometric implementation and synthetic tests. Ordinary scans remain `UNRESOLVED` without validated anatomical references and an observed scalp surface |
| G/O selection | Optional clicks in captured images only; never required for capture; remain 2D unless registered camera/surface association succeeds |
| Distances | Euclidean distance and Dijkstra mesh-edge shortest paths, explicitly labeled approximations; missing references make measurements unavailable |

The first engineering milestone has been demonstrated with **real COLMAP
reconstruction of synthetic multi-view geometry**, including dense stereo,
point cloud, mesh, camera poses, dimensions, and measurement output. This is
not evidence of real-head accuracy. The anatomical milestone currently provides
automatic facial/hairline observations, missing-reference handling, and tested
Kang geometry on known synthetic surfaces. It does **not** demonstrate reliable
automatic Kang localization on a real person. Real webcam scans, annotated
anatomical evaluation, and independently measured physical accuracy validation
remain required. No webcam was available in the implementation environment.

## Build and launch

Use the Ubuntu Docker/X11/XWayland and webcam setup in [README.md](README.md).
In the same desktop terminal, retain your UID/GID, camera group, and X11 setup:

```bash
cd /home/musa/Desktop/farFinder
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export VIDEO_GID="$(stat -c '%g' /dev/video0)"
./scripts/prepare-x11.sh
docker compose build
```

The original application still runs with:

```bash
docker compose up
docker compose down
```

Close the original camera application before starting the head interface:

```bash
docker compose run --rm farfinder python -m farfinder --head
```

Alternatively:

```bash
docker compose run --rm farfinder python -m farfinder.head live
```

To make head mode the application's default, add this service command to Compose:

```yaml
command: ["python", "-m", "farfinder", "--head"]
```

After that, `docker compose up` opens the head interface. Removing the command
returns to the original distance mode. Keep the default resolution at 640×480
initially; use higher resolution only after verifying camera/CPU performance.

## One-time intrinsic calibration

Distance calibration based on face size is **not intrinsic calibration**. The
intrinsic file stores `camera_matrix` with fx=`K[0][0]`, fy=`K[1][1]`,
cx=`K[0][2]`, cy=`K[1][2]`, distortion `[k1,k2,p1,p2,k3]`, reprojection RMS,
per-view errors, board specification, camera path, and actual camera resolution.

Generate the standard 7×5 square board, using 2 cm squares and 0.7 marker ratio:

```bash
docker run --rm --network none \
  -v farfinder_farfinder-data:/app/data \
  farfinder:local python -m farfinder.head board
docker run --rm --network none \
  -v farfinder_farfinder-data:/app/data:ro \
  farfinder:local cat /app/data/charuco_board/charuco.pdf > charuco.pdf
```

Print `charuco.pdf` at **100% / actual size**, without “fit to page.” Mount it
flat. Check multiple horizontal and vertical squares with a ruler; reject
nonuniform scaling or a warped board. No individual corner clicks are needed.

Start the head interface, press **C**, and move the board across the image while
varying tilt, distance, and position. The application automatically collects at
least 20 diverse views with at least 12 detected corners. Duplicate or blurred
views are rejected. It requires image-position coverage, RMS ≤1.5 px, per-view
errors ≤2 px, and plausible focal lengths. A failed calibration preserves the
previous file. Successful calibration saves `/app/data/intrinsics.json`.

Standalone calibration also works:

```bash
docker compose run --rm farfinder python -m farfinder.head calibrate
```

The default physical board specification is shared by capture and calibration.
For a different square size, use `board --square-cm VALUE` when generating, and
`calibrate --board-square-cm VALUE` when calibrating. Keep the board pattern and
marker ratio unchanged. Device path/resolution changes invalidate the intrinsic
file; do not silently resize a camera matrix. Recalibrate after changing focus,
zoom, crop, or lens settings even if the resolution remains unchanged.

## Capture workflow

1. Keep **one person stationary**. The person should hold the head still, keep a
   consistent expression, and avoid blinking or hair movement during adjacent
   views where possible. Use stable diffuse lighting.
2. Press **S / Start capture**. Move the **camera** slowly around the head, keeping
   the entire head in view. Do not rotate the subject to imitate a camera orbit.
3. Cover Front, Front-left, Left, Back-left, Back, Back-right, Right, Front-right,
   and Top. Take overlapping intermediate views as you rise toward the top;
   avoid jumping directly from a side view to a steep top view.
4. The application saves frames automatically. It rejects bad exposure, blur,
   clipped heads, insufficient texture, insufficient feature overlap, and
   redundant views. If overlap is lost, move back toward the last accepted view.
5. Press **T / Stop capture**, then **R / Reconstruct**. A session is limited to
   100 saved frames. A useful complete orbit typically has 40–80 overlapping
   frames, rather than only nine isolated views.

The local multi-class selfie model segments hair, face skin, and adjacent body
skin. Its largest head component supplies the mask. It is not a validated scalp
or ear segmentation model. A brief side/rear-view dropout can use **estimated
temporal head tracking**, initialized automatically from a clear hair/face view.
It requires consistent forward/backward optical flow, an inlier-supported motion
transform, and current hair/skin foreground evidence. It never accepts a warped
old mask without current foreground support. Tracked masks are tinted amber and
their source/flow quality is saved in frame metadata. Tracking expires after
30 frames or 3 seconds without a new reliable semantic anchor; return toward a
clear front view to reacquire. A new capture session resets tracking.

Bald rear views without visible face/hair beyond this short bridge can still be
unresolved and rejected. Other occlusions or atypical views can also fail;
missing coverage is reported rather than silently filled.

During capture, “head orientation” is estimated from visible facial geometry;
back/top views can have no facial orientation estimate. Absolute named-view
coverage is evaluated after SfM from camera centers relative to a frontal
anchor, with uncertainty noted. Camera poses are absent until registered.
Capture coverage cannot prove surface completeness or subject stationarity.

The original 30/60 cm distance estimate continues to display when a frontal face
passes its original quality checks. The original calibration is copied into the
session for provenance; intrinsic calibration never overwrites it.

## Reconstruction and quality gates

### Fixed webcam: experimental head-turn capture

If the webcam cannot move, use the separate head-turn mode:

```bash
docker compose run --rm farfinder python -m farfinder --head --capture-mode head-rotation
```

The moving-camera workflow remains the default. A rotating rigid object can
produce usable object-relative viewpoints; a human head is not guaranteed to
remain rigid. Head turns are an explicit experimental capture assumption,
not a certification that subject rotation is equivalent to a stationary-person
scan. No physical camera motion is inferred from facial pose estimates.
COLMAP's [official capture guidance](https://colmap.github.io/tutorial.html)
requires textured, overlapping views; these conditions alone do not validate
human-head rigidity or anatomical accuracy.

1. Press **S** while looking straight into the webcam. Keep your mouth closed
   and a neutral expression; avoid talking, blinking during a saved frame,
   moving hair, or moving your shoulders into the head mask.
2. Turn your head slowly toward one side, return through the front, and turn
   toward the other side. Use small overlapping changes, pausing briefly at
   each view. Keep the full head in frame, sharply focused and evenly lit.
   Aim for **30–60 accepted frames** across front and side views.
3. Watch **Saved yaw span**. Reconstruction requires at least 12 saved frames,
   at least six facial orientation estimates, a frontal anchor, turns of at
   least 15 degrees to each side, and at least 40 degrees total saved yaw span.
   This checks coarse capture diversity; it does not prove rigidity or 3D
   accuracy. Head-only matching and geometric verification still apply, with
   at least 65% geometric inliers required for adjacent saved head-turn views.
4. Press **T**, then **R**. If geometry succeeds, use **D**, **M**, and **W**.
   Reconstruction can still fail because of insufficient texture, feature
   overlap, inconsistent head geometry, poor calibration, or incomplete views.

Only surfaces actually visible in the captured images can be reconstructed.
Neck turns alone usually cannot cover the entire posterior head or scalp top.
Missing surfaces are not generated; measurements needing them are unavailable.

Session metadata and `camera_poses.json` explicitly mark poses as
`head_relative`. These describe COLMAP's effective viewpoints relative to the
approximately rigid head, not movement of the physical webcam. Existing
camera intrinsics and the frontal webcam-to-face distance feature can be reused.

**Metric scale stays unresolved in head-turn mode.** A stationary ChArUco board
and the turning head do not share the same object-relative motion, so the app
does not use that board to certify centimeters for this reconstruction. The
30/60 cm face-distance calibration also cannot certify its scale. A separately
validated co-moving reference method would be required; it is not currently
implemented. Intrinsic calibration with the printed board remains useful.

The synthetic rotating-target test keeps the physical camera and textured
background fixed, moves only a textured rigid ellipsoid, applies head masks,
and runs real COLMAP and stereo. Ground-truth object motion is withheld from
the reconstruction. This is a pipeline test, not anatomical or real-head
accuracy evaluation:

```bash
docker run --rm --network none \
  -v farfinder-head-validation:/app/data \
  farfinder:local python -m farfinder.head demo --capture-mode head-rotation
```

### Reconstruction stages

COLMAP 3.8 runs local CLI feature extraction, exhaustive matching with geometric
verification, and incremental SfM. All SIFT keypoints are limited by the head
masks; background features are not used to determine the head reconstruction.
Calibrated `FULL_OPENCV` intrinsics are supplied and held fixed during bundle
adjustment. Five OpenCV distortion coefficients are preserved.

A run needs at least six registered views, ≥65% of the captured views registered,
and at least 150 reliable head points with multi-view tracks and bounded
reprojection error. Points must agree with head masks in multiple views and in
the majority of registered camera views. Disconnected models, registration
fraction, reprojection error, mask support, view coverage, and surface detail
are recorded. A failure produces a clear error and quality report. A new run
invalidates old published landmarks/measurements/scale before processing, so a
failed rerun cannot leave stale centimeter measurements marked valid.

The supplied [COLMAP build is CPU-only](https://packages.debian.org/bookworm/colmap).
[COLMAP dense stereo requires CUDA](https://colmap.github.io/legacy/3.8/faq.html#available-functionality-without-gpu-cuda).
The application therefore provides **OpenCV CPU stereo** from the validated
camera poses: overlapping pair selection, stereo rectification, horizontal or
vertical SGBM, left/right consistency, texture filtering, head-mask agreement,
and support near reliable SfM geometry. It reports accepted pairs and retained
points. If dense stereo is insufficient, it explicitly reports a sparse-surface
fallback and applies the same observed-surface checks.

Open3D removes outliers, estimates normals, and performs ball pivoting. Long
triangle bridges and tiny disconnected fragments are removed. No Poisson hole
filling, learned canonical head, or hidden scalp/ear coordinates are added.
The result can be incomplete or disconnected. Hair/skin without stable visual
texture may make reconstruction fail entirely.

`--dense-backend cuda` contains the COLMAP dense-stage interface, but the supplied
Debian image has no CUDA support. It reports that limitation and uses CPU stereo.
Running that CUDA stage would require a CUDA-enabled COLMAP image plus an
explicitly exposed compatible GPU. GPU access and privileged mode are not
required by this project.

The surface is the **observed outer hair/skin surface**, not the skull or the
scalp hidden beneath hair. This limits both anatomical dimensions and Kang
geometry. Camera coverage alone cannot certify a complete head surface.

## Valid metric scale

The existing monocular 30/60 cm face calibration does **not automatically resolve
the similarity-scale ambiguity of multi-view SfM**. The same is true of knowing
fx/fy: camera intrinsics alone do not determine reconstruction scale.

The supported physical reference is a **static ChArUco board** fixed in the scene
relative to the stationary subject. Put it beside the head, outside the head
mask, but visible in several overlapping original images. Keep it stationary
throughout the capture. It is intentionally excluded from head feature matching;
its automatically detected corners are registered using the recovered cameras.

Independently measure a printed square and estimate that measurement's
uncertainty. For example, only if your actual ruler measurement was 2.00 cm
with uncertainty 0.01 cm:

```bash
docker compose run --rm farfinder python -m farfinder --head \
  --reference-square-cm 2.00 --reference-uncertainty-cm 0.01
```

Do not copy those values without measuring your own board. Numerical entry of a
physical reference dimension is allowed; no anatomical point selection is used.
The board must have at least eight corners each triangulated in at least three
registered views with adequate baseline and reprojection consistency. A rigid
similarity fit checks board geometry; residual and scale uncertainty thresholds
reject inconsistent results, including many board/subject-motion failures.
The scale certificate records its source, factor, reference count, and geometric
uncertainty proxy. It does not constitute a physical accuracy evaluation.

Without a valid certificate, all available distances use
**`reconstruction_units`**, and the interface states that scale is unresolved.
Point cloud, mesh, and camera coordinates remain in reconstruction units even
when metric scale is valid; `cm_per_unit` in the report supplies the conversion.
There is no fixed average head size, guessed interpupillary distance, or direct
conversion of MediaPipe relative Z into centimeters.

## Anatomy, Kang geometry, and optional inputs

MediaPipe supports facial landmarks. Glabella is currently an **estimated region
between eyebrow references**, and eyebrow peaks are estimated from eyebrow
landmarks after accounting for head roll. These are orientation/reference
estimates; they are not replacements for anatomical tragus tips or ear roots.

2D observations become 3D references only after they agree in at least three
registered camera views with adequate triangulation baseline, small
reprojection residuals, and an association with the observed surface. Unrelated
2D pixels are not averaged into 3D. Frontal hairline candidates are the same
mesh vertices verified against hair/skin boundary masks and ray visibility in
multiple camera views; this is still an unvalidated hairline estimate.

The four Kang definitions used by the geometric module are the definitions in
the project request, referring to [Lee et al. 2011](https://pubmed.ncbi.nlm.nih.gov/21585598/)
and [the paper DOI](https://doi.org/10.1111/j.1524-4725.2011.02020.x). Bibliographic
information was verified; the complete published headband/tapeline protocol
was not independently reproduced from its full text.

- **V:** superior observed scalp intersection of midsagittal and bilateral-tragus
  coronal references.
- **P:** posterior observed midsagittal intersection at the horizontal level of
  the bilateral superior ear attachments.
- **FL/FR:** frontal hairline intersections with the observed surface path from
  the geometrically established headband pole to the respective eyebrow peak.
  The current tape-path implementation is a mesh-edge approximation.

The coordinate frame uses validated bilateral anatomical ear references and
registered facial orientation: subject-left, superior, anterior. Bilateral
labels and the anterior direction are checked. Missing ears, unknown poles,
unregistered hairline, inconsistent geometry, absent surface intersections,
or an unvalidated scalp surface produce **`UNRESOLVED` with a reason**. Current
ordinary scans do not certify the reconstructed outer hair mesh as scalp;
that gate deliberately remains false. V/P are never assigned generic top/back
Face Mesh points. G/O are never relabeled as V/P.

**B / Optional G/O** opens captured images, without mirroring their coordinates:

- **G:** optional glabella point between the eyebrows.
- **O:** optional posterior skull/base reference point.
- **N/P:** next/previous captured image; **Q/Esc:** return to the live interface.

Choose G or O and click only if you want that reference. Neither click is required
to capture. A click remains a 2D observation unless its registered camera ray
hits the observed mesh. No UI or session API permits manual ear, V, P, FL, FR,
or other individual anatomical landmark placement.

## Local custom ear detector interface

No freely available model was identified and validated for the exact four
required ear references, so this application does not invent them. The default
`EarDetector` returns missing coordinates and reasons. A future local custom
YOLO keypoint model can use `YoloOnnxEarDetector` with OpenCV DNN CPU inference;
the runtime does not require PyTorch or Ultralytics accounts/services.

The supported ONNX output is decoded, one-class YOLO pose `[1,17,N]` or
`[1,N,17]`: bbox xywh, object score, then four `(x,y,confidence)` keypoints in
the exact manifest order. It assumes RGB 0–1 letterboxed input. NMS and keypoint
visibility/confidence gates are applied. See
[the manifest template](examples/ear-model-manifest.example.json).

Provide your own trained weights, exact semantic labels, model SHA-256, and an
annotated evaluation report with matching checksum. Without evaluation evidence,
predictions cannot supply validated anatomy to Kang geometry. The interface's
minimum evidence check is a provenance gate, not proof of medical accuracy.
Left/right tragus tips and superior roots need separately verified annotations
across frontal/profile/rear views and held-out subjects before reliance.

Copy local assets into the data volume, or add explicit read-only mounts, then:

```bash
docker compose run --rm farfinder python -m farfinder --head \
  --ear-model /app/data/models/ears.onnx \
  --ear-manifest /app/data/models/ears.json
```

`HeadbandPoleDetector` is a separate automatic-detector contract. It remains
unresolved until physical headband pole locations can be established by a
validated local marker/keypoint method. Do not estimate a pole solely from a
generic head shape. `calculate_kang` accepts validated references and a validated
observed scalp surface; its synthetic tests demonstrate that geometric interface,
not production anatomical localization.

## Measurements and interface controls

| Control | Action |
| --- | --- |
| S / Start capture | New automatic multi-view session |
| T / Stop capture | Stop saving frames |
| R / Reconstruct | COLMAP, dense stereo, observed point cloud/mesh and quality report |
| D / Detect landmarks | Local facial/hairline/optional ear detection and 3D registration |
| M / Measurements | Recalculate both distance modes and availability |
| E / Euclidean | Show straight 3D distances |
| U / Surface paths | Show mesh-edge Dijkstra approximations |
| W / Save results | Save session state; processing stages also save results automatically |
| C / Intrinsics | Automatic ChArUco intrinsic calibration |
| B / Optional G/O | Browse captured images for permitted optional selections |
| Arrow keys | Rotate the 3D preview |
| Q / Esc | Quit and close logs/resources |

The available measurement schema includes G→O, left ear→right ear, V→P,
V→left ear, V→right ear, FL→FR, and anatomical head width/height/length.
Missing anatomical endpoints make the corresponding measurement unavailable.
Head dimensions additionally require an anatomical coordinate frame, full
orbit/top camera coverage, and observed vertices near both required surface
extrema. They are labeled **anatomical axis extents of outer hair/skin**; skull
dimensions and surface completeness are not guaranteed. Axis extents are
distinct from diagonal point-to-point Euclidean distances.

Straight distances use registered 3D coordinates. Surface paths follow existing
mesh edges with Dijkstra and endpoint snapping; they are **approximations of
continuous mesh geodesics**, with unknown discretization error. Disconnected
endpoints or missing triangles produce unavailable distances, not a path across
an invented gap. Surface paths between dimension-extrema are separate from
the straight axis extent.

The interface marks `DETECTED`, `ESTIMATED`, and `UNRESOLVED`. Geometric
uncertainty, detector score proxies, and scale uncertainty are recorded/displayed
where available. They do not include all unknown anatomical, hair, calibration,
or model error and are not validated confidence intervals. Numerical centimeter
results require the independent scale certificate; centimeter **accuracy**
still requires a separately measured physical ground truth.

## Session files and batch commands

All sessions persist in the existing Docker volume under:

```text
/app/data/head_sessions/<session_id>/
  images/                   original, unmirrored PNG frames
  masks/                    COLMAP head-only masks
  segmentation/             local category masks
  session.json              frame metadata/calibration/optional G/O observations
  pointcloud.ply            observed head points, when reconstruction succeeds
  head_mesh.ply             observed triangles, when surface checks succeed
  camera_poses.json          registered K/R/t/distortion and coordinate convention
  landmarks.json            2D observations, 3D references, visibility/status/reasons
  measurements.json
  measurements.csv
  quality_report.json
  colmap.log                local command/error details
  colmap/                   per-run databases and intermediate models
```

Failed runs retain evidence and error reports. They do not create fake mesh or
point-cloud contents. Existing geometry from a prior run is refused when the
current report is FAILED/RUNNING. Fresh unavailable landmark/measurement records
invalidate stale results at the start of a new reconstruction.

Each measurement includes `measurement_name`, `start_landmark`, `end_landmark`,
`distance_value`, `distance_unit`, `distance_type`, `metric_scale_valid`,
`confidence_or_uncertainty`, `quality_status`, and a reason. Poses use
`X_camera = R_world_to_camera @ X_world + t_world_to_camera`.

To run without a webcam after capture, replace SESSION with the actual folder:

```bash
docker run --rm --network none -v farfinder_farfinder-data:/app/data \
  farfinder:local python -m farfinder.head reconstruct /app/data/head_sessions/SESSION
docker run --rm --network none -v farfinder_farfinder-data:/app/data \
  farfinder:local python -m farfinder.head detect /app/data/head_sessions/SESSION
docker run --rm --network none -v farfinder_farfinder-data:/app/data \
  farfinder:local python -m farfinder.head measure /app/data/head_sessions/SESSION
```

Add `--intrinsics /app/data/intrinsics.json` to reconstruction if the captured
session did not yet have intrinsics, and add both physical-reference arguments
when the independently measured static board appeared in that session.

Keep the same single user/volume ownership setup as the original app. Only one
writer should use a session at a time. `docker compose down` preserves the volume;
`docker compose down -v` deletes it. Head images are saved locally by design;
they are never transmitted, and retaining/removing them is your choice.

## Validation commands

Fast tests include all original Step A tests plus intrinsic calibration, camera
transforms, multi-view registration, Euclidean distances, curved mesh paths,
metric scale and moving-board rejection, missing anatomical landmarks, Kang
planes/tape paths, optional G/O restrictions, session serialization, capture
quality, native local models, and CPU mesh rendering:

```bash
docker compose config --quiet
docker compose build
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=128m,mode=1777 \
  farfinder:local python -m pip check
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=128m,mode=1777 \
  farfinder:local python -m pytest -q -p no:cacheprovider
```

Run the **full real COLMAP integration test** as well:

```bash
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=256m,mode=1777 \
  -e FARFINDER_RUN_RECONSTRUCTION_TEST=1 \
  farfinder:local python -m pytest -q -p no:cacheprovider
```

For a persistent demonstration with no webcam or desktop dependency:

```bash
docker run --rm --network none \
  -v farfinder_farfinder-data:/app/data \
  farfinder:local python -m farfinder.head demo
```

The demo renders a stationary textured ellipsoid from an orbit plus overlapping
top and bottom views, then performs real COLMAP feature extraction, matching,
geometric verification, and SfM. It receives no ground-truth poses or mesh for
reconstruction. Withheld synthetic poses/geometry are used only afterward to
evaluate registration, surface-radius residuals, and extent errors. Dense stereo
and Open3D process the actual reconstruction. Measurements remain in arbitrary
units, because a synthetic dimension is not an independently measured physical
centimeter reference. Ground-truth anatomical geometry is tested separately;
it never enables automatic clinical detection.

Implemented gates require at least 65% of synthetic cameras to register,
median relative surface-radius residual ≤15%, and axis extent deviations ≤10%.
These are engineering failure gates on this textured test object, not advertised
real-human measurement accuracy. Follow with a stationary textured physical
object and independently measured dimensions before scanning people. Annotated
ear/scalp/hairline ground truth is required before claiming anatomical accuracy.

The bundled models are downloaded only during image build and verified by
SHA-256. The additional model is
[MediaPipe multi-class selfie segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter),
float32 v1, checksum
`c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0`.
The project retains the GUI-enabled OpenCV wheel and pins Open3D's CPU wheel
and its full dependency tree; no competing OpenCV wheel is installed. Open3D
supports [Python 3.11 on Linux](https://www.open3d.org/docs/0.19.0/getting_started.html).
Compose retains a non-root user, read-only root filesystem, dropped capabilities,
no new privileges, offline networking, camera/X11 mounts, and persistent data.
