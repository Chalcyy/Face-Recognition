"""
app.py — native desktop dashboard (Tkinter) for the multi-face recognition +
expression pipeline in main.py. Pure Python UI, no browser/server needed.

Runs the SAME detection/tracking/vote-smoothing/expression logic as main.py's
`run()` loop, rendered inside a lavender/purple Tkinter window: live camera
feed on the left, a scrolling panel of detected people + their mood on the
right.

Usage:
  python app.py
  python app.py --max-faces 15
  python app.py --source path/to/video.mp4
"""

import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from main import (
    MODEL_PATH, RECOGNITION_THRESHOLD, RECOGNIZE_EVERY_N, MAX_FACES,
    FaceDatabase, IdentityLogger,
    landmarks_to_face_box, get_face_embeddings, make_landmarker,
    draw_face_box, _match_boxes_to_tracked, age_out_lost_faces,
    FACE_RECOGNITION_AVAILABLE,
)
import mediapipe as mp
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────────
BG          = "#16111f"
PANEL       = "#201a30"
PANEL_LINE  = "#34294f"
LAVENDER    = "#b9a4e8"
LAVENDER_2  = "#8f7ad6"
VIOLET      = "#6c4fc9"
PINK_GLOW   = "#f0a6c8"
TEXT_HI     = "#f2ecff"
TEXT_LO     = "#a496c4"
GOOD        = "#8fe3b0"
UNKNOWN_CLR = "#f0a6c8"

MOOD_EMOJI = {
    "Happy": "😊", "Sad": "😔", "Surprised": "😲",
    "Angry": "😠", "Disgusted": "😖", "Neutral": "😐",
}


class AuraApp:
    def __init__(self, root, source=0, model_path=MODEL_PATH,
                 threshold=RECOGNITION_THRESHOLD, max_faces=MAX_FACES,
                 enable_log=True):
        self.root = root
        self.source = source
        self.threshold = threshold
        self.max_faces = max_faces

        root.title("Aura — Live Presence Dashboard")
        root.configure(bg=BG)
        root.geometry("1180x680")
        root.minsize(900, 560)

        self.db = FaceDatabase()
        self.id_logger = IdentityLogger() if enable_log else None

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.lmk = make_landmarker(model_path, max_faces=max_faces)

        self.tracked_faces = []
        self.fc = 0
        self.prev_t = time.time()
        self.fps = 0.0

        self._build_ui()
        self._tick()

    # ── UI construction ────────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Frame(self.root, bg=BG, height=64)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", padx=24, pady=10)
        mark = tk.Canvas(brand, width=30, height=30, bg=BG, highlightthickness=0)
        mark.pack(side="left", padx=(0, 10))
        mark.create_rectangle(2, 2, 28, 28, fill=LAVENDER, outline="")
        title_box = tk.Frame(brand, bg=BG)
        title_box.pack(side="left")
        tk.Label(title_box, text="Aura", font=("Segoe UI", 15, "bold"),
                 fg=TEXT_HI, bg=BG).pack(anchor="w")
        tk.Label(title_box, text="Live presence & mood dashboard",
                 font=("Segoe UI", 8), fg=TEXT_LO, bg=BG).pack(anchor="w")

        status_box = tk.Frame(header, bg=PANEL, highlightbackground=PANEL_LINE,
                               highlightthickness=1)
        status_box.pack(side="right", padx=24, pady=14)
        self.status_dot = tk.Canvas(status_box, width=10, height=10, bg=PANEL,
                                     highlightthickness=0)
        self.status_dot.pack(side="left", padx=(10, 6), pady=6)
        self.status_dot.create_oval(1, 1, 9, 9, fill=GOOD, outline="")
        self.status_label = tk.Label(status_box, text="Live", font=("Segoe UI", 9),
                                      fg=TEXT_LO, bg=PANEL)
        self.status_label.pack(side="left", padx=(0, 12), pady=6)

        tk.Frame(self.root, bg=PANEL_LINE, height=1).pack(fill="x")

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=24, pady=20)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=340)
        main.rowconfigure(0, weight=1)

        # ── Feed panel ──
        feed_panel = tk.Frame(main, bg=PANEL, highlightbackground=PANEL_LINE,
                               highlightthickness=1)
        feed_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 20))

        feed_head = tk.Frame(feed_panel, bg=PANEL)
        feed_head.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(feed_head, text="Live Feed", font=("Segoe UI", 11, "bold"),
                 fg=TEXT_HI, bg=PANEL).pack(side="left")
        self.fps_badge = tk.Label(feed_head, text="-- FPS", font=("Segoe UI", 9),
                                   fg=LAVENDER, bg=PANEL)
        self.fps_badge.pack(side="right")

        self.video_label = tk.Label(feed_panel, bg="#0e0a17")
        self.video_label.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        legend = tk.Frame(feed_panel, bg=PANEL)
        legend.pack(fill="x", padx=16, pady=(0, 14))
        self._legend_item(legend, GOOD, "Known face")
        self._legend_item(legend, UNKNOWN_CLR, "Unknown face")

        # ── Side panel ──
        side = tk.Frame(main, bg=BG)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)

        summary = tk.Frame(side, bg=PANEL, highlightbackground=PANEL_LINE,
                            highlightthickness=1)
        summary.pack(fill="x", pady=(0, 16))
        self.stat_total = self._stat(summary, "In frame", TEXT_HI)
        tk.Frame(summary, bg=PANEL_LINE, width=1).pack(side="left", fill="y", pady=14)
        self.stat_known = self._stat(summary, "Known", GOOD)
        tk.Frame(summary, bg=PANEL_LINE, width=1).pack(side="left", fill="y", pady=14)
        self.stat_unknown = self._stat(summary, "Unknown", UNKNOWN_CLR)

        people_panel = tk.Frame(side, bg=PANEL, highlightbackground=PANEL_LINE,
                                 highlightthickness=1)
        people_panel.pack(fill="both", expand=True)
        tk.Label(people_panel, text="Detected People", font=("Segoe UI", 11, "bold"),
                 fg=TEXT_HI, bg=PANEL).pack(anchor="w", padx=16, pady=(14, 8))

        list_container = tk.Frame(people_panel, bg=PANEL)
        list_container.pack(fill="both", expand=True, padx=10, pady=(0, 14))

        canvas = tk.Canvas(list_container, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.people_frame = tk.Frame(canvas, bg=PANEL)
        self.people_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.people_frame, anchor="nw",
                              width=300)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.empty_label = tk.Label(
            self.people_frame,
            text="◇\n\nNo faces in view yet.\nStep into frame to appear here.",
            font=("Segoe UI", 9), fg=TEXT_LO, bg=PANEL, justify="center"
        )
        self.empty_label.pack(pady=40)

        footer = tk.Label(self.root, text="Aura runs entirely on-device — no data leaves this machine.",
                           font=("Segoe UI", 8), fg=TEXT_LO, bg=BG)
        footer.pack(pady=(0, 10))

    def _legend_item(self, parent, color, text):
        item = tk.Frame(parent, bg=PANEL)
        item.pack(side="left", padx=(0, 18))
        sw = tk.Canvas(item, width=10, height=10, bg=PANEL, highlightthickness=0)
        sw.pack(side="left", padx=(0, 6))
        sw.create_rectangle(0, 0, 10, 10, fill=color, outline="")
        tk.Label(item, text=text, font=("Segoe UI", 8), fg=TEXT_LO, bg=PANEL).pack(side="left")

    def _stat(self, parent, label, color):
        box = tk.Frame(parent, bg=PANEL)
        box.pack(side="left", expand=True, fill="x", pady=14)
        n = tk.Label(box, text="0", font=("Segoe UI", 18, "bold"), fg=color, bg=PANEL)
        n.pack()
        tk.Label(box, text=label.upper(), font=("Segoe UI", 7), fg=TEXT_LO, bg=PANEL).pack()
        return n

    # ── Per-person card rendering ──────────────────────────────────────
    def _render_people(self, faces):
        for child in self.people_frame.winfo_children():
            child.destroy()

        if not faces:
            self.empty_label = tk.Label(
                self.people_frame,
                text="◇\n\nNo faces in view yet.\nStep into frame to appear here.",
                font=("Segoe UI", 9), fg=TEXT_LO, bg=PANEL, justify="center"
            )
            self.empty_label.pack(pady=40)
            return

        for face in faces:
            known = face.known
            name = face.name if known else "Unknown"
            mood = face.expression or "Neutral"
            emoji = MOOD_EMOJI.get(mood, "😐")
            conf_pct = round(face.confidence * 100)

            card = tk.Frame(self.people_frame, bg="#241d38",
                             highlightbackground=(PANEL_LINE if known else "#5a3550"),
                             highlightthickness=1)
            card.pack(fill="x", pady=5, padx=2)

            avatar_bg = LAVENDER_2 if known else UNKNOWN_CLR
            avatar = tk.Canvas(card, width=38, height=38, bg="#241d38", highlightthickness=0)
            avatar.pack(side="left", padx=10, pady=10)
            avatar.create_rectangle(0, 0, 38, 38, fill=avatar_bg, outline="")
            initials = "".join(w[0].upper() for w in name.split()[:2]) if known else "?"
            avatar.create_text(19, 19, text=initials, fill=BG, font=("Segoe UI", 11, "bold"))

            info = tk.Frame(card, bg="#241d38")
            info.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=8)

            tk.Label(info, text=name, font=("Segoe UI", 10, "bold"),
                     fg=TEXT_HI, bg="#241d38", anchor="w").pack(fill="x")

            meta = tk.Frame(info, bg="#241d38")
            meta.pack(fill="x", pady=(2, 4))
            mood_chip = tk.Label(meta, text=f" {emoji} {mood} ", font=("Segoe UI", 8, "bold"),
                                  fg=LAVENDER, bg="#2e2447")
            mood_chip.pack(side="left")
            conf_text = f"{conf_pct}% match" if known else "No match"
            tk.Label(meta, text=f"  {conf_text}", font=("Segoe UI", 8),
                     fg=TEXT_LO, bg="#241d38").pack(side="left")

            bar_track = tk.Canvas(info, height=4, bg=PANEL_LINE, highlightthickness=0)
            bar_track.pack(fill="x")
            bar_track.update_idletasks()

            def draw_bar(canvas=bar_track, pct=(conf_pct if known else 100), color=(LAVENDER if known else UNKNOWN_CLR)):
                canvas.delete("all")
                w = canvas.winfo_width() or 260
                fill_w = max(int(w * pct / 100), 4)
                canvas.create_rectangle(0, 0, fill_w, 4, fill=color, outline="")
            card.after(10, draw_bar)

    # ── Main loop ───────────────────────────────────────────────────────
    def _tick(self):
        ok, frame = self.cap.read()
        if ok:
            if isinstance(self.source, int):
                frame = cv2.flip(frame, 1)

            self.fc += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.lmk.detect(mp_img)

            if result and result.face_landmarks:
                boxes = [landmarks_to_face_box(lm, w, h) for lm in result.face_landmarks]
                blendshapes = (result.face_blendshapes if result.face_blendshapes
                               else [None] * len(boxes))
                self.tracked_faces = _match_boxes_to_tracked(boxes, self.tracked_faces, blendshapes)

                if self.fc % RECOGNIZE_EVERY_N == 0:
                    embeddings = get_face_embeddings(
                        rgb, [f.box for f in self.tracked_faces], num_jitters=1
                    )
                    for face, emb in zip(self.tracked_faces, embeddings):
                        if emb is not None:
                            raw_name, raw_conf = self.db.identify(emb, threshold=self.threshold)
                        else:
                            raw_name, raw_conf = "Unknown", 0.0
                        face.register_vote(raw_name, raw_conf)
                        if self.id_logger and face.name != "Unknown":
                            self.id_logger.log(face.name, face.confidence)

                for face in self.tracked_faces:
                    draw_face_box(frame, face.box, face.name,
                                  face.confidence, face.known, face.expression)
            else:
                self.tracked_faces = age_out_lost_faces(self.tracked_faces)

            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.prev_t, 1e-6))
            self.prev_t = now

            # ── push frame into the video Label ──
            disp_w = max(self.video_label.winfo_width(), 480)
            disp_h = max(self.video_label.winfo_height(), 270)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((disp_w, disp_h))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # ── update stats + side panel ──
            known = sum(1 for f in self.tracked_faces if f.known)
            total = len(self.tracked_faces)
            self.stat_total.configure(text=str(total))
            self.stat_known.configure(text=str(known))
            self.stat_unknown.configure(text=str(total - known))
            self.fps_badge.configure(text=f"{self.fps:.0f} FPS")
            self._render_people(self.tracked_faces)

        self.root.after(30, self._tick)

    def close(self):
        self.cap.release()
        self.lmk.close()
        if self.id_logger:
            self.id_logger.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Aura — native Tkinter dashboard")
    p.add_argument("--source", default=0)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD)
    p.add_argument("--max-faces", type=int, default=MAX_FACES)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    if not FACE_RECOGNITION_AVAILABLE:
        raise SystemExit("face_recognition is not installed. Run: pip install face_recognition")
    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")

    src = args.source
    try:
        src = int(src)
    except (TypeError, ValueError):
        pass

    root = tk.Tk()
    app = AuraApp(root, source=src, model_path=args.model, threshold=args.threshold,
                  max_faces=args.max_faces, enable_log=not args.no_log)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.close(), root.destroy()))
    root.mainloop()