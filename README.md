# farFinder

Measure approximate webcam-to-face distance in centimeters with a live OpenCV
window, MediaPipe facial landmarks, FPS, calibration status, and measurement CSVs.
Python 3.11, OpenCV, MediaPipe, NumPy, and the tests run **inside Docker**. No Python
packages need to be installed on the Ubuntu host.

Processing uses the local CPU. No account, API key, paid API, or cloud inference is
required. The image downloads a public model **during build**, verifies its SHA-256,
and bundles it. After building, the application runs offline; Compose explicitly
disables container networking. No images or video are saved.

## Ubuntu prerequisites

- Ubuntu on an **x86_64/amd64** computer with an X11 desktop, or Wayland with
  **XWayland** enabled. This pinned MediaPipe Linux wheel targets amd64; ARM Linux
  is outside the supported configuration.
- Docker Engine and the Docker Compose v2 plugin. Follow the official
  [Docker installation instructions for Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
  Confirm `docker version` and `docker compose version` work. Docker daemon access
  must already be configured for your user; run desktop setup as your normal user.
- A working V4L2 webcam at `/dev/video0`, and a ruler or tape measure.
- The Ubuntu system utility `xauth` for desktop authorization:

  ```bash
  sudo apt update
  sudo apt install xauth
  ```

The initial build needs internet access to download the base image, OS libraries,
Python wheels, and the model. Subsequent starts need no network.

## Build and run

Open a terminal **in your Ubuntu desktop session**, change into this project,
and prepare the camera group and X11 cookie:

```bash
cd /path/to/farFinder
ls -l /dev/video0
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export VIDEO_GID="$(stat -c '%g' /dev/video0)"
./scripts/prepare-x11.sh
docker compose build
docker compose up
```

Keep these exports in the same terminal for Compose commands. The image's
non-root user is built with your UID/GID, so it can read the mode-600 X11 cookie.
The supplemental `VIDEO_GID` gives it access to the mapped camera. The default
UID/GID is 1000 and the default camera group is 44, but the exports above avoid
assuming those values. Rebuild when changing `LOCAL_UID` or `LOCAL_GID`.

The setup script copies only your current display's authorization cookie into
`.docker.xauth`, ignored by Docker and Git, and changes its address family to
permit the container hostname. Compose mounts the cookie and X11 Unix socket
read-only. Re-run the script after logging out/in or changing `DISPLAY`. Keep the
cookie private. A local display such as `:0` or `:1` is supported; SSH forwarding
and a desktop without XWayland are outside this setup.

Close the window, press **Q** or **Esc**, or press **Ctrl+C** in the Compose terminal
to stop. Remove the stopped container with:

```bash
docker compose down
```

`docker compose down` preserves the data volume. **Do not add `--volumes` or `-v`**
unless you intend to delete calibration and measurement history. To start again:

```bash
docker compose up
```

## Two-point calibration

Calibrate for **one person, one camera, and one camera resolution**. Sit directly
in front of the lens, keep your whole face visible, look straight at the camera,
and use steady lighting. Hold the ruler beside your face without covering it.
Use the same reference for both points: **camera lens to the tip of your nose**.

1. Position your nose **30 cm** from the lens, press **1**, and hold still until
   the window reports that 30 cm was captured.
2. Position your nose **60 cm** from the lens, press **2**, and hold still until
   the window reports that calibration was saved.
3. Move between the points to see the distance update. A green reading indicates
   accepted measurements; the quality panel explains rejected frames or
   extrapolation. Check against your ruler before relying on the estimate.

Each point needs at least **45 consecutive accepted frames spanning 1.5 seconds**
with eye-span relative standard deviation at most **3%**. Bad frames clear the
current batch; a large movement restarts it. A capture times out after 15 seconds
and can be retried. The 60 cm capture requires a completed 30 cm capture.
Implausible point pairs are rejected. The previous saved calibration is replaced
only after a successful new pair; distance display is suspended while collecting
a new pair.

| Key | Action |
| --- | --- |
| **1** | Start a new calibration by capturing 30 cm |
| **2** | Capture 60 cm after a successful 30 cm capture |
| **R** | Reset saved calibration; retain CSV history |
| **Q / Esc** | Quit and close the measurement CSV |

A valid saved calibration loads automatically on restart. The application refuses
calibration from a different device path or actual frame size. Recalibrate for
another person, a different webcam on the same path, or any change to zoom, camera
crop, position, or optics; those changes cannot all be detected automatically.

## Measurement and quality

MediaPipe's [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
detects landmarks locally. The feature is the pixel distance between the **outer
eye corners, landmarks 33 and 263**, rather than the model's relative Z coordinate.
Two measured points fit an inverse-size model:

```text
distance_cm = scale / outer_eye_span_px + offset_cm
```

The calibration determines both coefficients without assuming a universal face
size. Frames must have exactly one detected face, finite landmarks, an unclipped
face, at least a 40-pixel eye span, and a usable pose matrix. The application
rejects yaw or pitch above 20 degrees and roll above 25 degrees, dark or
overexposed face regions, and blurred frames. MediaPipe detection, presence, and
tracking thresholds are 0.6; they gate detection internally rather than serving
as a reported accuracy score.

A five-reading median filter followed by an exponential moving average
(`alpha = 0.35`) reduces jitter and isolated spikes. Rejected frames immediately
clear the distance and filter state, so a previous estimate cannot linger as a
current measurement. FPS reflects completed processing frames. The mirrored
preview shows the landmarks and eye-span line; mirroring happens after inference.

The calibrated interval is **30–60 cm**. Estimates outside it are marked
**extrapolated**; estimates outside 20–120 cm are suppressed. Accuracy depends on
the ruler measurements, camera geometry, lighting, head position, and landmark
stability. This is an approximate monocular measurement: remain centered and
frontal, and expect some lag from smoothing. Quality checks cannot guarantee
accuracy or reliably detect every occlusion.

## Persistent files

Compose mounts the named volume **`farfinder_farfinder-data`** at `/app/data`.
The container writes:

- `calibration.json`: current coefficients, point medians, camera, actual frame
  size, feature/schema version, calibration ID, and UTC creation time. Written
  atomically and checked when loading.
- `calibration.csv`: append-only history with two rows per completed calibration,
  including distances, spans, sample counts, stability, and coefficients.
- `measurements_<UTC timestamp>_<session id>.csv`: one file per run, up to 10 rows
  per second, with UTC time, elapsed time, raw and smoothed distance, eye span,
  FPS, quality, pose angles, calibration ID, and whether the reading is inside
  the calibrated interval. Invalid distances are blank, with a quality reason.

Measurements flush at least once per second and on a normal exit. A forced kill
or power loss may lose the most recent buffered rows. Only run one instance
against this volume at a time.

Copy files to the host **before `docker compose down`** (a stopped container after
Ctrl+C is also suitable):

```bash
docker compose cp farfinder:/app/data ./data
```

## Automated tests and Docker validation

Tests do not require a webcam, display server, X11 cookie, or internet. Build the
image, validate Compose, check dependency consistency, and run the tests:

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

The suite covers two-point fitting and interpolation, invalid calibration and
camera changes, stable capture and timeout, smoothing and rejected frames,
atomic persistence, CSV logging, geometry, pose, lighting/blur rejection, camera
failure cleanup, the keyboard calibration workflow, preview rendering, and real
inference against a blank frame with the bundled model.
The model smoke test runs without networking. Live camera access and GUI display
still need a manual check on the target desktop: start the application, complete
both calibration points, confirm measurement against a ruler, then restart and
confirm calibration reloads.

## Configuration and troubleshooting

The default capture request is 640×480 at 30 FPS; the webcam may negotiate a
different size. The actual size is logged and stored in the calibration. To change
the request, add a service command in `docker-compose.yml` and recalibrate:

```yaml
command: ["python", "-m", "farfinder", "--width", "1280", "--height", "720", "--fps", "30"]
```

For another webcam, change the host side of the mapping (for example
`/dev/video2:/dev/video0:rw`) and export that device's group as `VIDEO_GID`.
The application continues to use `/dev/video0` inside the container; reset and
recalibrate when switching cameras.

- **Missing `/dev/video0`:** connect the webcam and check `ls -l /dev/video*`.
  Compose requires the mapped device to exist. Some webcams expose additional
  metadata-only nodes; select the actual video capture node.
- **Cannot open camera:** close other camera applications and verify
  `VIDEO_GID="$(stat -c '%g' /dev/video0)"`. The device is mapped with read/write
  access. Do not use `privileged: true` or change its permissions to world-writable.
- **X11 authorization / Qt xcb error:** verify `echo "$DISPLAY"`, run the setup
  script again in your desktop terminal, and rebuild with your UID/GID exports.
  `XAUTHORITY` may point to an XWayland cookie under `/run/user/...`; the script
  respects it. Do not disable X11 access control with `xhost +`.
- **Mount source missing:** run `./scripts/prepare-x11.sh` before `up`. Compose
  deliberately fails when the cookie or X11 socket directory is absent instead
  of creating an incorrect directory mount.
- **Data permission denied after changing user:** an existing volume retains its
  original ownership. Return to the original build UID/GID, or back up and migrate
  its files before changing ownership. Rebuilding does not change existing volume
  ownership.
- **Calibration rejected / timed out:** check the ruler reference, keep the face
  frontal and fully visible, improve lighting, focus the webcam, and retry **1**
  then **2**. Increasing capture resolution can help a small eye span.
- **Low FPS:** request a smaller resolution, close CPU-heavy programs, and keep
  only one camera application open.

## Container and model details

The image pins Python 3.11.14, MediaPipe 0.10.21, the GUI-enabled
`opencv-contrib-python` 4.11.0.86, NumPy 1.26.4, and all Python transitive/test
dependencies in `requirements.txt`. The contrib package supplies `cv2` and also
satisfies MediaPipe's dependency; installing another OpenCV wheel alongside it
would risk losing GUI support. The Dockerfile checks dependency compatibility.

The model is the versioned public MediaPipe Face Landmarker float16 v1 bundle:

- [Model file](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- [Model documentation and model card](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models)

Compose uses a non-root image user, drops all Linux capabilities, prohibits new
privileges, sets a read-only root filesystem, and provides writable `/tmp` and
`/app/data`. It maps only the requested webcam and the desktop socket/cookie.
It exposes no ports, uses no GPU, and requires neither a privileged container nor
host networking.
