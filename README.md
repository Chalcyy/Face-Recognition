# Real-Time Multi-Face Recognition + Expression Detection

Detects **every face visible in the camera at once** using MediaPipe, and
for each one:
- identifies **who** it is against a persistent, on-disk database of
  enrolled people (name + confidence, or `Unknown`), with vote-smoothing so
  the label doesn't flicker frame to frame,
- reads off their **facial expression** (Happy, Sad, Surprised, Angry,
  Disgusted, or Neutral) from the same landmark pass, at no extra cost.

All of this runs independently per face — with several people in frame,
each gets their own box, name, and expression label simultaneously.

## How it works

1. **Detection** — MediaPipe `FaceLandmarker` runs with `num_faces=N`
   (configurable, default **`10`**) so it locates landmarks for up to `N`
   faces in every frame, not just the first one it sees.
2. **Boxing** — Each face's landmark cloud is converted into a tight
   `(top, right, bottom, left)` bounding box, reusing MediaPipe's landmarks
   instead of running a second face detector.
3. **Tracking faces across frames** — Before anything else, each frame's
   boxes are matched to the previous frame's tracked faces by nearest
   box-center (`_match_boxes_to_tracked`). This is what lets identity and
   expression follow the right physical person as they move, instead of
   labels jumping between faces or resetting every frame.
4. **Embedding** — `face_recognition` (dlib ResNet, ~99.4% LFW accuracy)
   computes a 128-d identity embedding for **every** tracked face's box in
   one batched call.
5. **Matching** — Each embedding is compared against everyone in
   `face_data/known_faces.pkl`. The closest enrolled sample under
   `RECOGNITION_THRESHOLD` (default `0.55`) is that face's raw candidate
   identity for this pass; otherwise it's `Unknown`.
6. **Stable labeling (majority vote)** — Recognition (steps 4–5) is the
   expensive step, so it only runs every `RECOGNIZE_EVERY_N` frames (default
   `8`). Each raw result is added to that face's rolling history
   (`VOTE_HISTORY_LEN` = 5 results). The **displayed** name only changes once
   a candidate wins a clear majority (`VOTE_MIN_AGREEMENT` = 3 of 5) — so one
   bad frame (blur, angle, lighting) can't flip a known face to `Unknown`
   and back.
7. **Expression detection** — MediaPipe's `FaceLandmarker` also outputs 52
   ARKit-style blendshape scores per face when `output_face_blendshapes=True`
   is set. `detect_expression()` maps combinations of those scores (e.g.
   `mouthSmileLeft/Right` → Happy, `jawOpen + browInnerUp + eyeWide` →
   Surprised) to a label. This reuses the same landmark pass, so it updates
   **every frame** — no extra model, no extra cost, and no need for vote
   smoothing the way identity has.

Green box = known face, red box = unknown face. The name/confidence label
sits below the box; the expression label sits above it.

## Setup

```bash
pip install -r requirements.txt
```

You also need the MediaPipe face landmark model file, `face_landmarker.task`,
in the same folder as `main.py`. Download it with:

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

> **Note on `dlib`/`face_recognition`:** these can be slow to install (dlib
> compiles from source on most platforms). On Windows, installing via
> `pip install dlib` may require CMake and a C++ build toolchain (Visual
> Studio Build Tools). On macOS/Linux, `cmake` and a C compiler are usually
> enough (`brew install cmake` / `apt install cmake build-essential`).

## Usage

**Enroll someone** (captures 20 samples by default from different angles):
```bash
python main.py --enroll "Alice" --samples 20
```
> Enrollment purposefully locks onto a single (the largest/closest) face
> per frame, even in multi-face mode, so you don't accidentally enroll
> someone else who wanders into the shot.

**Run live multi-face recognition + expression detection:**
```bash
python main.py
```

**Change how many faces can be tracked at once** (default is 10):
```bash
python main.py --max-faces 15
```

**List everyone currently enrolled:**
```bash
python main.py --list
```

**Remove someone from the database:**
```bash
python main.py --remove "Alice"
```

**Other useful flags:**
```bash
python main.py --source path/to/video.mp4     # run on a video file instead of a webcam
python main.py --threshold 0.5                # stricter identity matching (fewer false positives)
python main.py --no-log                       # disable identity_log.csv writing
```

Press **ESC** or **Q** at any time to quit a running window.

## Files & folders

```
main.py                       # this script
face_landmarker.task          # MediaPipe model (download separately, see Setup)
requirements.txt
face_data/
  known_faces.pkl             # enrolled identities + embeddings (auto-created)
  identity_log.csv            # timestamped log of who was seen, when (auto-created)
```

`face_data/` is created automatically on first run and persists across
restarts — enroll once, and recognition keeps working in future sessions.
Enrolling more samples per person over time (different lighting, angles,
expressions) improves matching accuracy. Deleting `identity_log.csv` doesn't
disable logging — it just gets recreated fresh next run unless you pass
`--no-log`.

## Desktop Dashboard (pure Python, no browser)

`app.py` is a native desktop dashboard built with **Tkinter** (Python's
built-in GUI toolkit) + Pillow for rendering frames — no Flask, no server,
no HTML, no browser tab.

```bash
python app.py
```

This opens a native window: live feed on the left, detected people + mood
panel on the right, with a known/unknown summary bar on top. It reuses the
identical detection/tracking/vote-smoothing/expression pipeline from
`main.py`. Same tuning flags apply:

```bash
python app.py --max-faces 15
python app.py --source path/to/video.mp4
python app.py --no-log
```



| Setting | Where | Effect |
|---|---|---|
| `--max-faces` / `MAX_FACES` | CLI flag / top of `main.py` | Max simultaneous faces detected & recognized per frame (default `10`) |
| `--threshold` / `RECOGNITION_THRESHOLD` | CLI flag / top of `main.py` | Lower = stricter identity matching, fewer false "known" hits |
| `RECOGNIZE_EVERY_N` | top of `main.py` | Higher = better FPS, slower to first-recognize a newly-appeared face |
| `VOTE_HISTORY_LEN` | top of `main.py` | How many recent recognition passes are remembered per face (default `9`) |
| `VOTE_MIN_AGREEMENT` | top of `main.py` | How many of those must agree before the shown label switches (default `6`) — raise for more stability, lower to relabel faster |
| `MISSED_FRAMES_GRACE` | top of `main.py` | Frames a face can briefly go undetected (blink, quick head turn) before it's dropped and has to re-earn its label from scratch (default `10`) |
| `MATCH_MAX_DIST` | top of `main.py` | Max pixel distance a box can move between frames and still count as the same tracked face |
| `EXPRESSION_RULES` | top of `main.py` | Blendshape combos + thresholds defining each expression label — lower a threshold to trigger that expression more easily |
| `--samples` | CLI flag | More enrollment samples = more robust identity matching per person |

## Troubleshooting

- **"Model not found"** — download `face_landmarker.task` into the same
  folder as `main.py` (see Setup).
- **"face_recognition is not installed"** — `pip install face_recognition`
  (requires `dlib`, see the setup note above).
- **Names still flicker between `Unknown` and a person** — raise
  `VOTE_HISTORY_LEN` (e.g. `7`) and `VOTE_MIN_AGREEMENT` (e.g. `4`) for
  stronger smoothing, or increase `--samples` when enrolling for a more
  robust average embedding per person.
- **Laggy / low FPS with several faces** — raise `RECOGNIZE_EVERY_N` (e.g.
  `12`–`15`) or lower `--max-faces`.
- **Only some faces detected in a crowded frame** — raise `--max-faces`
  (MediaPipe won't report more faces than `num_faces` allows), or lower
  `min_face_detection_confidence` in `make_landmarker()` if faces are
  small/far/at an angle.
- **Expression always shows Neutral, or the wrong one** — lower the
  relevant threshold in `EXPRESSION_RULES`; blendshape scores vary with
  lighting and camera angle, so thresholds may need tuning per setup.
- **Expression looks right but attached to the wrong face** — this
  shouldn't happen since blendshapes are paired with each face's box before
  tracking reassigns them, but if faces overlap heavily or cross paths, try
  lowering `MATCH_MAX_DIST` so tracking doesn't hop to a nearby face.