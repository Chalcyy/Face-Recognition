# Face Recognition

Real-time webcam app that detects a face with MediaPipe and identifies **who**
it belongs to, using a persistent, on-disk database of enrolled face
embeddings.

## How it works

1. MediaPipe `FaceLandmarker` locates the face and its landmarks every frame
   (fast — this is the only detector running).
2. The landmark cloud is turned into a tight bounding box.
3. `face_recognition` (dlib's ResNet face encoder, ~99.4% accuracy on the LFW
   benchmark) computes a 128-d identity embedding for that box.
4. The embedding is compared against every sample stored in
   `face_data/known_faces.pkl`. The closest match under a distance threshold
   is reported as the identity, with a confidence score.

Recognition (step 3) only runs every `RECOGNIZE_EVERY_N` frames (default 5) —
it's the heavier step, so this keeps the feed smooth.

## Why data is stored

- **Persistence** — enrolled identities survive restarts; you don't need to
  re-enroll every session.
- **Improving accuracy** — enrolling more samples per person (different
  angles, lighting, expressions) makes future matching more robust. Re-run
  `--enroll` on the same name anytime to add more samples; they accumulate.
- **Future reference** — every recognition event is appended to
  `face_data/identity_log.csv` (timestamp, name, confidence) so you have a
  record of who was seen and when.

## Requirements

- Python 3.9–3.11
- A webcam (or a video file path via `--source`)
- MediaPipe's `face_landmarker.task` model file

```bash
pip install -r requirements.txt
```

### Installing `dlib` / `face_recognition`

`face_recognition` depends on `dlib`, a compiled C++ library.

- **Ubuntu / Debian**
  ```bash
  sudo apt-get install -y build-essential cmake
  pip install -r requirements.txt
  ```
- **macOS**
  ```bash
  brew install cmake
  pip install -r requirements.txt
  ```
- **Windows** — prefer a prebuilt wheel rather than compiling:
  ```bash
  pip install dlib-bin
  pip install face_recognition face-recognition-models
  ```

If `face_recognition` isn't installed, the app exits with a clear error
telling you to install it — recognition is the whole point of this app now.

### Downloading the MediaPipe model

```bash
curl -L -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Place it next to `main.py`, or pass a custom path with `--model`.

## Usage

**1. Enroll people** (build the database):
```bash
python main.py --enroll "Alice" --samples 20
```
Move your head slightly during capture — varied angles/expressions improve
matching. Run `--enroll "Alice"` again later to add more samples for the same
person.

**2. Run live recognition:**
```bash
python main.py
```
Each detected face gets a green box + name + confidence if recognized, or a
red box + "Unknown" if not.

**3. Manage the database:**
```bash
python main.py --list              # show everyone enrolled
python main.py --remove "Alice"    # delete a person
```

### CLI flags

| Flag             | Description                                         |
|-------------------|-----------------------------------------------------|
| `--source`        | Camera index (default `0`) or video file path       |
| `--model`         | Path to `face_landmarker.task`                       |
| `--threshold`     | Match strictness (lower = stricter, default `0.55`) |
| `--no-log`        | Disable identity CSV logging                         |
| `--enroll NAME`   | Enrollment mode instead of live recognition          |
| `--samples`       | Samples to capture during `--enroll` (default 20)    |
| `--list`          | List enrolled identities and exit                    |
| `--remove NAME`   | Remove a person from the database and exit           |

Press **ESC** or **Q** to quit, or close the window.

## Data layout

```
face_data/
├── known_faces.pkl     # {name: [embedding, embedding, ...]} — pickled database
└── identity_log.csv     # timestamp, name, confidence — recognition history
```

## Notes & privacy

- Everything stays local — nothing is uploaded anywhere.
- Treat `face_data/` like biometric data: don't commit it to a public repo,
  and only enroll people who've consented to being recognized.
- Recognition accuracy depends on enrollment quality — consistent lighting, a
  few head angles per person, and adding more samples over time all help.