"""
Real-Time Multi-Face Recognition
==================================
Detects ALL faces in the frame with MediaPipe and identifies WHO each face
belongs to using a persistent, on-disk database of enrolled face embeddings.

Pipeline (per frame):
  1. MediaPipe FaceLandmarker finds every face and its landmarks (num_faces=N).
  2. Each landmark cloud is turned into a tight bounding box (fast — no second
     detector needed).
  3. `face_recognition` (dlib ResNet encoder, ~99.4% LFW accuracy) computes a
     128-d identity embedding for each box.
  4. Each embedding is matched against everyone enrolled in
     face_data/known_faces.pkl. The closest match under a distance threshold
     is reported as that face's identity. Every face is labeled independently
     — known faces get a name + confidence, unknown faces get "Unknown".

Usage:
  python main.py --enroll "Alice" --samples 20   # register a new person
  python main.py                                  # live recognition (multi-face)
  python main.py --list                           # show enrolled people
  python main.py --remove "Alice"                 # delete a person
  python main.py --max-faces 10                   # detect up to 10 faces/frame
"""

import pickle
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

BaseOptions        = mp.tasks.BaseOptions
FaceLandmarker      = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts  = mp.tasks.vision.FaceLandmarkerOptions
RunningMode         = mp.tasks.vision.RunningMode

MODEL_PATH = "face_landmarker.task"

# ── Persistent storage ──────────────────────────────────────────────────────
# Embeddings are stored on disk so identities survive restarts, and matching
# gets more accurate the more samples are enrolled per person over time.
FACE_DATA_DIR          = Path("face_data")
FACE_DB_PATH           = FACE_DATA_DIR / "known_faces.pkl"
IDENTITY_LOG_PATH      = FACE_DATA_DIR / "identity_log.csv"
RECOGNITION_THRESHOLD  = 0.55   # lower = stricter match (face_distance units)
RECOGNIZE_EVERY_N      = 8      # run the (heavier) recognition model every N frames
MIN_SAMPLES_PER_PERSON = 20     # samples collected during --enroll
MAX_FACES              = 10     # max simultaneous faces MediaPipe will track

# ── Stability tuning ─────────────────────────────────────────────────────────
# A single recognition pass can be noisy (blur, angle, lighting) and flip a
# face between a name and "Unknown" from one pass to the next. To avoid that
# flicker, each tracked face keeps a short rolling history of raw recognition
# results and only changes its DISPLAYED label once a result wins a majority
# vote over that history -- so one bad frame can't flip the label by itself.
VOTE_HISTORY_LEN    = 9     # how many recent recognition results to remember per face
VOTE_MIN_AGREEMENT  = 6     # how many of those must agree before switching the shown label
MATCH_MAX_DIST       = 140.0 # px -- how far a box can move between passes and still count as "the same face"
MISSED_FRAMES_GRACE  = 10    # frames a face can go undetected before it's dropped (avoids reset on a blink/brief occlusion)

# ── Colours (BGR) ────────────────────────────────────────────────────────────
C = {
    "bg":       (18, 18, 24),
    "known":    (80, 210, 80),
    "unknown":  (60, 60, 200),
    "white":    (240, 240, 245),
    "grey":     (150, 150, 165),
    "accent":   (255, 180, 60),
}


# ══════════════════════════════════════════════════════════════════════════════
# Persistent face database
# ══════════════════════════════════════════════════════════════════════════════

class FaceDatabase:
    """
    Stores each enrolled person's face embeddings on disk (pickle) so that:
      - identities survive across program restarts (future reference), and
      - recognition accuracy improves over time -- every extra enrolled sample
        (different angle/lighting/expression) makes matching more robust.

    Layout: { "Alice": [emb1, emb2, ...], "Bob": [emb1, ...] }
    Each embedding is a 128-d numpy vector from the face_recognition model.
    """

    def __init__(self, db_path=FACE_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.load()

    def load(self):
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)
            n_people  = len(self.data)
            n_samples = sum(len(v) for v in self.data.values())
            print(f"[FaceDB] Loaded {n_people} identities, {n_samples} samples "
                  f"from {self.db_path}")
        else:
            self.data = {}
            print(f"[FaceDB] No existing database found -- starting fresh "
                  f"({self.db_path})")

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    def add_sample(self, name, embedding):
        self.data.setdefault(name, []).append(embedding)
        self.save()

    def remove_person(self, name):
        if name in self.data:
            del self.data[name]
            self.save()
            return True
        return False

    def all_embeddings(self):
        names, embs = [], []
        for name, samples in self.data.items():
            for e in samples:
                names.append(name)
                embs.append(e)
        return names, embs

    def identify(self, embedding, threshold=RECOGNITION_THRESHOLD):
        """Return (name, confidence in [0,1]). 'Unknown' if no sample is close enough."""
        names, embs = self.all_embeddings()
        if not embs:
            return "Unknown", 0.0
        dists      = face_recognition.face_distance(embs, embedding)
        best_idx   = int(np.argmin(dists))
        best_dist  = float(dists[best_idx])
        confidence = float(np.clip(1.0 - best_dist, 0.0, 1.0))
        if best_dist <= threshold:
            return names[best_idx], confidence
        return "Unknown", confidence

    def summary(self):
        if not self.data:
            return "empty"
        return ", ".join(f"{n} ({len(v)})" for n, v in self.data.items())


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe landmarks → face box → embedding
# ══════════════════════════════════════════════════════════════════════════════

def landmarks_to_face_box(landmarks, w, h, margin=0.25):
    """
    Derive a (top, right, bottom, left) box -- the format face_recognition
    expects -- from a MediaPipe face-landmark cloud. Reusing MediaPipe's
    already-computed landmarks means we skip running a second, slower face
    detector.
    """
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = max(0, int(x1 - mw)); y1 = max(0, int(y1 - mh))
    x2 = min(w, int(x2 + mw)); y2 = min(h, int(y2 + mh))
    return (y1, x2, y2, x1)   # top, right, bottom, left


def get_face_embeddings(rgb_frame, face_boxes, num_jitters=1):
    """
    Compute a 128-d identity embedding for EACH given face box in one batched
    call. Returns a list the same length as face_boxes (None for any box a
    valid embedding could not be produced for -- practically never once a
    box is already known-valid).
    """
    if not FACE_RECOGNITION_AVAILABLE or not face_boxes:
        return [None] * len(face_boxes)
    encodings = face_recognition.face_encodings(
        rgb_frame, known_face_locations=face_boxes, num_jitters=num_jitters
    )
    # face_recognition returns encodings in the same order as face_boxes.
    if len(encodings) != len(face_boxes):
        # Defensive fallback -- pad/truncate rather than crash the loop.
        encodings = (encodings + [None] * len(face_boxes))[:len(face_boxes)]
    return encodings


def make_landmarker(model_path, max_faces=MAX_FACES):
    opts = FaceLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,   # needed for expression detection
    )
    return FaceLandmarker.create_from_options(opts)


# ══════════════════════════════════════════════════════════════════════════════
# Expression detection (from MediaPipe blendshapes)
# ══════════════════════════════════════════════════════════════════════════════

# Each expression is defined as a set of blendshape names (MediaPipe's
# ARKit-style 52 blendshapes) whose scores we sum/average. This avoids
# needing a second model -- FaceLandmarker already computes these per face
# whenever output_face_blendshapes=True.
EXPRESSION_RULES = {
    "Happy":     (["mouthSmileLeft", "mouthSmileRight"], 0.35),
    "Sad":       (["mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight"], 0.25),
    "Surprised": (["jawOpen", "browInnerUp", "eyeWideLeft", "eyeWideRight"], 0.30),
    "Angry":     (["browDownLeft", "browDownRight", "noseSneerLeft", "noseSneerRight"], 0.30),
    "Disgusted": (["noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft", "mouthUpperUpRight"], 0.30),
}


def detect_expression(blendshapes):
    """
    Given a face's blendshape list (result.face_blendshapes[i]), score every
    rule in EXPRESSION_RULES and return (label, score). Falls back to
    "Neutral" if nothing crosses its threshold.
    """
    if not blendshapes:
        return "Neutral", 0.0

    scores = {b.category_name: b.score for b in blendshapes}

    best_label, best_score = "Neutral", 0.0
    for label, (names, threshold) in EXPRESSION_RULES.items():
        vals = [scores.get(n, 0.0) for n in names]
        avg  = sum(vals) / len(vals) if vals else 0.0
        if avg >= threshold and avg > best_score:
            best_label, best_score = label, avg
    return best_label, best_score


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

class IdentityLogger:
    """
    Appends every recognition event to a persistent CSV (face_data/identity_log.csv)
    so who-was-seen-when is kept for future reference / auditing. Uses append
    mode so history accumulates across every run. Tracks last-seen state
    PER NAME (not globally) so multiple simultaneous faces are each logged
    independently without spamming duplicate rows.
    """
    def __init__(self, path=IDENTITY_LOG_PATH):
        import csv
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new  = not self.path.exists()
        self.fh = open(self.path, "a", newline="")
        self.wr = csv.writer(self.fh)
        if is_new:
            self.wr.writerow(["timestamp", "name", "confidence"])
        self._last_time = {}   # name -> last logged timestamp

    def log(self, name, confidence):
        now = time.time()
        last = self._last_time.get(name, 0.0)
        # avoid spamming duplicate rows for the same person -- log at most
        # once every 5s per identity (each name tracked independently so
        # multiple people in frame simultaneously all get logged).
        if (now - last) < 5.0:
            return
        self._last_time[name] = now
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.wr.writerow([ts, name, f"{confidence:.3f}"])
        self.fh.flush()

    def close(self):
        self.fh.close()
        print(f"[IdentityLogger] Saved → {self.path}")


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════

def draw_face_box(frame, box, name, confidence, known, expression=None):
    top, right, bottom, left = box
    col = C["known"] if known else C["unknown"]
    cv2.rectangle(frame, (left, top), (right, bottom), col, 2)

    label = f"{name}  {confidence*100:.0f}%" if known else "Unknown"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (left, bottom), (left + tw + 16, bottom + th + 16), col, -1)
    cv2.putText(frame, label, (left + 8, bottom + th + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if expression:
        (ew, eh), _ = cv2.getTextSize(expression, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
        cv2.rectangle(frame, (left, top - eh - 14), (left + ew + 16, top), C["accent"], -1)
        cv2.putText(frame, expression, (left + 8, top - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(frame, fps, db_summary, n_known=0, n_unknown=0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), C["bg"], -1)
    cv2.putText(frame,
                f"Face Recognition   {fps:.0f} FPS   |   "
                f"Known: {n_known}   Unknown: {n_unknown}",
                (10, 23), cv2.FONT_HERSHEY_DUPLEX, 0.55, C["accent"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"DB: {db_summary}", (10, h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, C["grey"], 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Enrollment mode
# ══════════════════════════════════════════════════════════════════════════════

def enroll_person(name, source=0, model_path=MODEL_PATH,
                   num_samples=MIN_SAMPLES_PER_PERSON):
    """
    Opens the camera, tracks the (largest, closest) face with MediaPipe, and --
    whenever it's detected -- computes and stores a face_recognition embedding
    under `name`. Collecting several samples (different angles/expressions)
    is what makes later recognition more accurate and robust.

    NOTE: enrollment intentionally uses only ONE face per frame (the largest)
    even though live recognition supports many -- you want to be sure you're
    only enrolling the person sitting in front of the camera, not whoever
    else might wander into frame.
    """
    if not FACE_RECOGNITION_AVAILABLE:
        raise RuntimeError(
            "face_recognition is not installed. Run: pip install face_recognition"
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    db  = FaceDatabase()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 55)
    print(f"  ENROLLING '{name}'  |  collecting {num_samples} samples")
    print("  Look at the camera, slowly turn your head slightly.")
    print("  Press ESC or Q to cancel.")
    print("=" * 55)

    collected = 0
    fc = 0
    cv2.namedWindow("Enroll Face", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enroll Face", 960, 540)

    # Enrollment always tracks a single (largest) face regardless of MAX_FACES.
    with make_landmarker(model_path, max_faces=MAX_FACES) as lmk:
        while cap.isOpened() and collected < num_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)
            fc += 1
            h, w = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lmk.detect(mp_img)

            box_drawn = False
            if result and result.face_landmarks:
                # If several people are in frame during enrollment, pick the
                # largest face box (closest to camera) as the enrollment target.
                boxes = [landmarks_to_face_box(lm, w, h) for lm in result.face_landmarks]
                areas = [(b[2] - b[0]) * (b[1] - b[3]) for b in boxes]
                box   = boxes[int(np.argmax(areas))]

                if fc % 3 == 0:
                    emb = get_face_embeddings(rgb, [box], num_jitters=1)[0]
                    if emb is not None:
                        db.add_sample(name, emb)
                        collected += 1
                    top, right, bottom, left = box
                    cv2.rectangle(frame, (left, top), (right, bottom), C["known"], 2)
                    box_drawn = True

                if not box_drawn:
                    top, right, bottom, left = box
                    cv2.rectangle(frame, (left, top), (right, bottom), C["accent"], 1)

            cv2.rectangle(frame, (0, h - 40), (w, h), C["bg"], -1)
            cv2.putText(frame, f"Samples: {collected}/{num_samples}", (14, h - 14),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, C["accent"], 1, cv2.LINE_AA)

            cv2.imshow("Enroll Face", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if cv2.getWindowProperty("Enroll Face", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

    if collected > 0:
        print(f"[Enroll] Stored {collected} new samples for '{name}'.")
        print(f"[Enroll] Database now: {db.summary()}")
    else:
        print("[Enroll] No samples captured — nothing was saved.")


# ══════════════════════════════════════════════════════════════════════════════
# Live recognition loop (MULTI-FACE)
# ══════════════════════════════════════════════════════════════════════════════

def run(source=0, model_path=MODEL_PATH, threshold=RECOGNITION_THRESHOLD,
        enable_log=True, max_faces=MAX_FACES):
    if not FACE_RECOGNITION_AVAILABLE:
        raise RuntimeError(
            "face_recognition is not installed. Run: pip install face_recognition"
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    db = FaceDatabase()
    if not db.data:
        print("[WARN] No one is enrolled yet — every face will show as 'Unknown'.")
        print("       Enroll someone first: python main.py --enroll \"Name\"")

    id_logger = IdentityLogger() if enable_log else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 1280, 720)

    print("=" * 55)
    print("  LIVE Multi-Face Recognition  |  Press ESC or Q to quit")
    print(f"  Max simultaneous faces: {max_faces}")
    print(f"  Known identities: {db.summary()}")
    print("=" * 55)

    fps_q  = deque(maxlen=30)
    prev_t = time.time()
    fc     = 0

    # tracked_faces is a list of TrackedFace objects, one per currently-visible
    # face, persisted across frames (matched by position, not index) so each
    # face keeps its own rolling vote history and only relabels on majority
    # agreement -- this is what kills the flicker.
    tracked_faces = []

    with make_landmarker(model_path, max_faces=max_faces) as lmk:
        while cap.isOpened():
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                break
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)

            fc += 1
            h, w = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = lmk.detect(mp_img)

            if result and result.face_landmarks:
                # 1) Compute a box for EVERY detected face this frame.
                boxes = [landmarks_to_face_box(lm, w, h) for lm in result.face_landmarks]
                blendshapes_by_box = (
                    result.face_blendshapes if result.face_blendshapes
                    else [None] * len(boxes)
                )

                # 2) Re-associate this frame's boxes with existing tracked
                #    faces (by nearest box-center) BEFORE recognition, so
                #    vote history stays attached to the same physical face
                #    even as people move or the box jitters slightly.
                #    Blendshapes travel alongside their box so expression
                #    stays correctly paired with the right face too.
                tracked_faces = _match_boxes_to_tracked(
                    boxes, tracked_faces, blendshapes_by_box
                )

                # 3) Recognition is the heavier step -- only run it every N
                #    frames, batched across all faces at once. Each result
                #    is fed into that face's vote history rather than
                #    overwriting the displayed label directly.
                if fc % RECOGNIZE_EVERY_N == 0:
                    embeddings = get_face_embeddings(
                        rgb, [f.box for f in tracked_faces], num_jitters=1
                    )
                    for face, emb in zip(tracked_faces, embeddings):
                        if emb is not None:
                            raw_name, raw_conf = db.identify(emb, threshold=threshold)
                        else:
                            raw_name, raw_conf = "Unknown", 0.0
                        face.register_vote(raw_name, raw_conf)
                        if id_logger and face.name != "Unknown":
                            id_logger.log(face.name, face.confidence)

                # 4) Expression is set inside _match_boxes_to_tracked from
                #    the blendshapes paired with each box -- read straight
                #    from the landmarker's output, no extra model call, so
                #    it updates every frame (cheap) rather than being gated
                #    behind RECOGNIZE_EVERY_N like recognition is.

                for face in tracked_faces:
                    draw_face_box(frame, face.box, face.name,
                                  face.confidence, face.known, face.expression)
            else:
                tracked_faces = age_out_lost_faces(tracked_faces)

            n_known   = sum(1 for f in tracked_faces if f.known)
            n_unknown = len(tracked_faces) - n_known

            now = time.time()
            fps_q.append(1.0 / max(now - prev_t, 1e-6))
            prev_t = now
            draw_hud(frame, float(np.mean(fps_q)), db.summary(), n_known, n_unknown)

            cv2.imshow("Face Recognition", frame)

            elapsed_loop = time.time() - loop_start
            wait = max(1, int((0.05 - elapsed_loop) * 1000))
            key = cv2.waitKey(wait) & 0xFF
            if key in (27, ord('q')):
                break
            if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    if id_logger:
        id_logger.close()
    seen = ", ".join(f.name for f in tracked_faces) if tracked_faces else "none"
    print(f"\nLast faces seen: {seen}")


# ══════════════════════════════════════════════════════════════════════════════
# Stable per-face identity tracking (majority-vote smoothing)
# ══════════════════════════════════════════════════════════════════════════════

class TrackedFace:
    """
    Represents one physical face being tracked across frames. Keeps a short
    rolling history of raw per-pass recognition results and only changes the
    DISPLAYED name/known state when a candidate wins a clear majority over
    that history. This is what prevents a single bad frame (blur, angle,
    lighting) from flipping the label back and forth every recognition pass.
    """
    __slots__ = ("box", "name", "confidence", "known", "_history", "expression", "missed")

    def __init__(self, box):
        self.box        = box
        self.name        = "Unknown"
        self.confidence  = 0.0
        self.known       = False
        self._history    = deque(maxlen=VOTE_HISTORY_LEN)
        self.expression  = "Neutral"
        self.missed      = 0   # consecutive frames this face wasn't re-detected

    def register_vote(self, raw_name, raw_conf):
        self._history.append(raw_name)

        counts = Counter(self._history)
        top_name, top_count = counts.most_common(1)[0]

        # Only switch the displayed label once a candidate has a clear
        # majority (>= VOTE_MIN_AGREEMENT out of the recent history) --
        # otherwise keep showing whatever was stable before, so a single
        # stray "Unknown" (or a single stray wrong name) can't flip it.
        if top_count >= VOTE_MIN_AGREEMENT and top_name != self.name:
            self.name  = top_name
            self.known = top_name != "Unknown"

        # Confidence display always reflects the latest matched pass for
        # whichever name currently won the vote (freshest useful number),
        # falling back to the raw value if names disagree this round.
        if raw_name == self.name:
            self.confidence = raw_conf


def _box_center(box):
    top, right, bottom, left = box
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _match_boxes_to_tracked(boxes, tracked_faces, blendshapes_by_box=None, max_dist=MATCH_MAX_DIST):
    """
    Reassigns each new frame's boxes to existing TrackedFace objects by
    nearest box-center, so vote history (and therefore the stable label)
    stays attached to the same physical face as people move -- instead of
    assuming face order/index stays constant frame to frame. Boxes with no
    close previous match get a brand-new TrackedFace (starts as Unknown
    until enough recognition passes vote it in).

    blendshapes_by_box, if given, must be the same length/order as boxes --
    each face's expression is (re)computed here every frame straight from
    its blendshapes, since that's cheap and doesn't need the vote-smoothing
    recognition does.
    """
    used = set()
    result = []
    prev = [( _box_center(f.box), f) for f in tracked_faces]
    if blendshapes_by_box is None:
        blendshapes_by_box = [None] * len(boxes)

    for box, shapes in zip(boxes, blendshapes_by_box):
        c = _box_center(box)
        best_i, best_d = None, max_dist
        for i, (pc, f) in enumerate(prev):
            if i in used:
                continue
            d = ((c[0] - pc[0]) ** 2 + (c[1] - pc[1]) ** 2) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            face = prev[best_i][1]
            face.box = box   # refresh position, keep identity + vote history
            face.missed = 0
        else:
            face = TrackedFace(box)

        if shapes:
            face.expression, _ = detect_expression(shapes)

        result.append(face)
    return result


def age_out_lost_faces(tracked_faces):
    """
    Called on a frame where MediaPipe reports ZERO faces at all (e.g. a
    blink, brief motion blur, or someone turning their head for an instant).
    Instead of wiping every tracked face immediately -- which would cause
    the box/label to disappear and reappear as "new" (losing vote history
    and looking unstable) -- each face gets a few frames of grace
    (MISSED_FRAMES_GRACE) before being dropped for real.
    """
    survivors = []
    for face in tracked_faces:
        face.missed += 1
        if face.missed <= MISSED_FRAMES_GRACE:
            survivors.append(face)
    return survivors


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Multi-Face Recognition (MediaPipe + face_recognition)")
    p.add_argument("--source", default=0,
                    help="Camera index or video file path")
    p.add_argument("--model", default=MODEL_PATH,
                    help="Path to the MediaPipe face_landmarker.task model")
    p.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD,
                    help="Face match distance threshold (lower = stricter)")
    p.add_argument("--no-log", action="store_true",
                    help="Disable identity CSV logging")
    p.add_argument("--enroll", metavar="NAME", default=None,
                    help="Enrollment mode: capture face samples for NAME and exit")
    p.add_argument("--samples", type=int, default=MIN_SAMPLES_PER_PERSON,
                    help="Number of samples to capture during --enroll")
    p.add_argument("--list", action="store_true",
                    help="List enrolled identities and exit")
    p.add_argument("--remove", metavar="NAME", default=None,
                    help="Remove NAME from the database and exit")
    p.add_argument("--max-faces", type=int, default=MAX_FACES,
                    help="Maximum number of simultaneous faces to detect/recognize per frame")
    args = p.parse_args()

    src = args.source
    try:    src = int(src)
    except: pass

    try:
        if args.list:
            db = FaceDatabase()
            print(f"Enrolled identities: {db.summary()}")
        elif args.remove:
            db = FaceDatabase()
            ok = db.remove_person(args.remove)
            print(f"Removed '{args.remove}'." if ok else f"'{args.remove}' not found.")
        elif args.enroll:
            enroll_person(args.enroll, source=src, model_path=args.model,
                           num_samples=args.samples)
        else:
            run(source=src, model_path=args.model, threshold=args.threshold,
                enable_log=not args.no_log, max_faces=args.max_faces)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")