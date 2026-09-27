"""
Advanced Hand, Body & Face Tracker
==================================

A bigger companion to hand_body_detection.py, adding:

    1 - ASL Letter Recognition : train a simple nearest-neighbor
                                  classifier on YOUR OWN hand shapes and
                                  have it recognize static ASL letters
                                  live (note: motion letters J and Z
                                  can only be captured as one static pose)
    2 - Snake Game              : control a snake with your hand position
                                  (acts like a joystick) in a separate window
    3 - Pong Game                : control paddles with your hands (one
                                  hand per paddle) in a separate window
    0 - Plain detection          : just show hand/body/face landmarks

Also always available (independent of mode):
    f - toggle face landmark detection on/off
    r - toggle video recording (saves an .mp4 of what you see)
    l - toggle landmark CSV logging (saves numeric landmark data per frame)

Base detection uses:
    - OpenCV        : camera capture, display, drawing, video writing
    - MediaPipe      : HandLandmarker + PoseLandmarker + FaceLandmarker
                        (Tasks API - this is the modern API; mediapipe
                        1.0+ removed the old mp.solutions.* API)

Run:
    python advanced_tracker.py

Full controls:
    q         - quit
    h         - toggle hand landmark detection
    b         - toggle body (pose) landmark detection
    f         - toggle face landmark detection
    r         - toggle video recording
    l         - toggle landmark CSV logging
    0         - mode: plain detection
    1         - mode: ASL letter recognition
    2         - mode: Snake game
    3         - mode: Pong game
    (in ASL mode only)
    t         - toggle training / recognition sub-mode
    a-z       - while training, label the current hand shape with that
                letter and save it as a training sample
"""

import os
import csv
import time
import math
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
ASL_DATA_PATH = os.path.join(BASE_DIR, "asl_training_data.csv")

HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model(path, url, label):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {label} model (one-time download)...")
        urllib.request.urlretrieve(url, path)
        print(f"{label} model saved to {path}")


# -----------------------------------------------------------------------
# Landmark connection maps for manual drawing
# -----------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_landmarks(frame, landmarks, connections, point_color, line_color, radius=4):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], line_color, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), radius, point_color, -1)


def draw_face_points(frame, landmarks, color=(180, 180, 255)):
    h, w = frame.shape[:2]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 1, color, -1)


# -----------------------------------------------------------------------
# ASL letter recognition: simple nearest-neighbor classifier trained on
# hand shapes the USER records. No external dataset needed.
# -----------------------------------------------------------------------
def extract_hand_feature(hand_landmarks):
    """Turn 21 hand landmarks into a position/scale-invariant feature vector."""
    wrist = hand_landmarks[0]
    ref = hand_landmarks[9]  # middle finger MCP - used to normalize scale
    scale = math.dist((wrist.x, wrist.y, wrist.z), (ref.x, ref.y, ref.z))
    scale = scale if scale > 1e-6 else 1e-6
    feats = []
    for lm in hand_landmarks:
        feats.append((lm.x - wrist.x) / scale)
        feats.append((lm.y - wrist.y) / scale)
        feats.append((lm.z - wrist.z) / scale)
    return np.array(feats, dtype=np.float32)


class ASLClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels = []
        self.vectors = []
        self._load()

    def _load(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    self.labels.append(row[0])
                    self.vectors.append(np.array(row[1:], dtype=np.float32))
            print(f"Loaded {len(self.labels)} ASL training samples from {self.data_path}")

    def add_sample(self, label, vector):
        self.labels.append(label)
        self.vectors.append(vector)
        write_header = not os.path.exists(self.data_path)
        with open(self.data_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + list(vector))

    def predict(self, vector, k=3, max_distance=1.6):
        if not self.vectors:
            return None, 0
        dists = [np.linalg.norm(vector - v) for v in self.vectors]
        order = np.argsort(dists)[:k]
        if dists[order[0]] > max_distance:
            return None, 0
        votes = {}
        for i in order:
            votes[self.labels[i]] = votes.get(self.labels[i], 0) + 1
        best = max(votes, key=votes.get)
        confidence = votes[best] / len(order)
        return best, confidence

    def sample_counts(self):
        counts = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts


# -----------------------------------------------------------------------
# Snake game - controlled like a joystick by hand position relative to
# the center of the camera frame
# -----------------------------------------------------------------------
class SnakeGame:
    GRID = 20
    CELL = 20  # canvas is GRID*CELL pixels square

    def __init__(self):
        self.reset()

    def reset(self):
        c = self.GRID // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self.direction = (1, 0)
        self.pending_direction = (1, 0)
        self.spawn_food()
        self.score = 0
        self.game_over = False
        self.last_tick = time.time()
        self.tick_interval = 0.15

    def spawn_food(self):
        while True:
            pos = (np.random.randint(0, self.GRID), np.random.randint(0, self.GRID))
            if pos not in self.snake:
                self.food = pos
                return

    def set_direction(self, dx, dy):
        # ignore reversing directly into itself
        if (dx, dy) != (-self.direction[0], -self.direction[1]):
            self.pending_direction = (dx, dy)

    def update(self):
        if self.game_over:
            return
        now = time.time()
        if now - self.last_tick < self.tick_interval:
            return
        self.last_tick = now
        self.direction = self.pending_direction
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        # wrap around the edges instead of ending the game
        new_head = ((head_x + dx) % self.GRID, (head_y + dy) % self.GRID)

        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

    def draw(self):
        size = self.GRID * self.CELL
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        fx, fy = self.food
        cv2.rectangle(canvas, (fx * self.CELL, fy * self.CELL),
                      ((fx + 1) * self.CELL, (fy + 1) * self.CELL), (0, 0, 255), -1)
        for i, (sx, sy) in enumerate(self.snake):
            color = (0, 255, 0) if i > 0 else (0, 255, 255)
            cv2.rectangle(canvas, (sx * self.CELL, sy * self.CELL),
                          ((sx + 1) * self.CELL, (sy + 1) * self.CELL), color, -1)
        cv2.putText(canvas, f"Score: {self.score}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.game_over:
            cv2.putText(canvas, "GAME OVER", (size // 2 - 100, size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(canvas, "press 'g' to restart", (size // 2 - 120, size // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return canvas


# -----------------------------------------------------------------------
# Pong game - two paddles, one controlled per hand
# -----------------------------------------------------------------------
class PongGame:
    WIDTH, HEIGHT = 480, 320
    PADDLE_W, PADDLE_H = 12, 70
    BALL_R = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self.left_y = self.HEIGHT / 2
        self.right_y = self.HEIGHT / 2
        self.ball = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        angle = np.random.uniform(-0.4, 0.4)
        direction = np.random.choice([-1, 1])
        self.ball_vel = np.array([direction * 260 * math.cos(angle), 260 * math.sin(angle)], dtype=np.float32)
        self.score_left = 0
        self.score_right = 0
        self.last_update = time.time()

    def set_paddles(self, left_norm_y, right_norm_y):
        if left_norm_y is not None:
            self.left_y = float(np.clip(left_norm_y, 0, 1)) * self.HEIGHT
        if right_norm_y is not None:
            self.right_y = float(np.clip(right_norm_y, 0, 1)) * self.HEIGHT

    def update(self):
        now = time.time()
        dt = min(now - self.last_update, 0.05)
        self.last_update = now

        self.ball += self.ball_vel * dt

        if self.ball[1] <= self.BALL_R or self.ball[1] >= self.HEIGHT - self.BALL_R:
            self.ball_vel[1] *= -1

        # left paddle collision
        if self.ball[0] - self.BALL_R <= self.PADDLE_W and \
                abs(self.ball[1] - self.left_y) <= self.PADDLE_H / 2:
            self.ball_vel[0] = abs(self.ball_vel[0])
        # right paddle collision
        if self.ball[0] + self.BALL_R >= self.WIDTH - self.PADDLE_W and \
                abs(self.ball[1] - self.right_y) <= self.PADDLE_H / 2:
            self.ball_vel[0] = -abs(self.ball_vel[0])

        if self.ball[0] < 0:
            self.score_right += 1
            self._reset_ball()
        elif self.ball[0] > self.WIDTH:
            self.score_left += 1
            self._reset_ball()

    def _reset_ball(self):
        self.ball = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        angle = np.random.uniform(-0.4, 0.4)
        direction = np.random.choice([-1, 1])
        self.ball_vel = np.array([direction * 260 * math.cos(angle), 260 * math.sin(angle)], dtype=np.float32)

    def draw(self):
        canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        cv2.line(canvas, (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT), (60, 60, 60), 2)
        cv2.rectangle(canvas, (0, int(self.left_y - self.PADDLE_H / 2)),
                      (self.PADDLE_W, int(self.left_y + self.PADDLE_H / 2)), (0, 255, 255), -1)
        cv2.rectangle(canvas, (self.WIDTH - self.PADDLE_W, int(self.right_y - self.PADDLE_H / 2)),
                      (self.WIDTH, int(self.right_y + self.PADDLE_H / 2)), (0, 255, 0), -1)
        cv2.circle(canvas, (int(self.ball[0]), int(self.ball[1])), self.BALL_R, (255, 255, 255), -1)
        cv2.putText(canvas, f"{self.score_left}  -  {self.score_right}",
                    (self.WIDTH // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return canvas


# -----------------------------------------------------------------------
# Landmark CSV logger
# -----------------------------------------------------------------------
class LandmarkLogger:
    HAND_COLS = [f"hand{h}_{i}_{axis}" for h in (1, 2) for i in range(21) for axis in "xyz"]
    POSE_COLS = [f"pose_{i}_{axis}" for i in range(33) for axis in "xyz"]

    def __init__(self, path):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp"] + self.HAND_COLS + self.POSE_COLS)

    def log(self, hand_result, pose_result):
        row = [time.time()]
        hand_data = [""] * (21 * 3 * 2)
        if hand_result and hand_result.hand_landmarks:
            for h_idx, lm_list in enumerate(hand_result.hand_landmarks[:2]):
                base = h_idx * 21 * 3
                for i, lm in enumerate(lm_list):
                    hand_data[base + i * 3: base + i * 3 + 3] = [lm.x, lm.y, lm.z]
        row.extend(hand_data)

        pose_data = [""] * (33 * 3)
        if pose_result and pose_result.pose_landmarks:
            lm_list = pose_result.pose_landmarks[0]
            for i, lm in enumerate(lm_list):
                pose_data[i * 3: i * 3 + 3] = [lm.x, lm.y, lm.z]
        row.extend(pose_data)

        self.writer.writerow(row)

    def close(self):
        self.file.close()


def hand_zone_direction(landmarks, frame_w, frame_h, deadzone=0.12):
    """Return a (dx, dy) joystick-style direction from an index fingertip
    position relative to the frame center. Used for the Snake game."""
    tip = landmarks[8]
    dx_norm = tip.x - 0.5
    dy_norm = tip.y - 0.5
    if abs(dx_norm) < deadzone and abs(dy_norm) < deadzone:
        return None
    if abs(dx_norm) > abs(dy_norm):
        return (1, 0) if dx_norm > 0 else (-1, 0)
    else:
        return (0, 1) if dy_norm > 0 else (0, -1)


def main():
    ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker")
    ensure_model(POSE_MODEL_PATH, POSE_MODEL_URL, "pose landmarker")
    ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarker")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    BaseOptions = mp_python.BaseOptions
    VisionRunningMode = mp_vision.RunningMode

    hand_landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detect_hands, detect_body, detect_face = True, True, False
    mode = 0
    mode_names = {0: "Plain Detection", 1: "ASL Recognition", 2: "Snake Game", 3: "Pong Game"}

    asl = ASLClassifier(ASL_DATA_PATH)
    asl_training = False

    snake = SnakeGame()
    pong = PongGame()

    video_writer = None
    recording = False
    logger = None
    logging_on = False

    prev_time = 0
    start_time = time.time()

    print("Advanced tracker running. Press 'q' to quit.")
    print("Modes: 0=plain 1=ASL 2=Snake 3=Pong | f=face r=record l=log")

    while True:
        success, frame = cap.read()
        if not success:
            print("Error reading from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms) if detect_body else None
        if pose_result and pose_result.pose_landmarks:
            for pl in pose_result.pose_landmarks:
                draw_landmarks(frame, pl, POSE_CONNECTIONS, (245, 117, 66), (245, 66, 230))

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms) if detect_hands else None
        if hand_result and hand_result.hand_landmarks:
            for hl, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
                draw_landmarks(frame, hl, HAND_CONNECTIONS, (0, 255, 255), (0, 200, 0))
                wrist = hl[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                raw_label = handed[0].category_name
                label = "Right" if raw_label == "Left" else "Left"
                cv2.putText(frame, label, (cx - 20, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if detect_face:
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if face_result and face_result.face_landmarks:
                for fl in face_result.face_landmarks:
                    draw_face_points(frame, fl)

        # -----------------------------------------------------------
        # MODE 1: ASL Recognition
        # -----------------------------------------------------------
        if mode == 1:
            sub = "TRAINING (press a letter key to save a sample)" if asl_training else "RECOGNIZING"
            cv2.putText(frame, f"ASL mode: {sub}  (t = switch)", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if hand_result and hand_result.hand_landmarks:
                vec = extract_hand_feature(hand_result.hand_landmarks[0])
                if not asl_training:
                    label, conf = asl.predict(vec)
                    text = f"{label} ({conf * 100:.0f}%)" if label else "Unknown / not enough samples yet"
                    cv2.putText(frame, text, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
                else:
                    counts = asl.sample_counts()
                    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no samples yet"
                    cv2.putText(frame, summary[:90], (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Show one hand to the camera", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # -----------------------------------------------------------
        # MODE 2: Snake
        # -----------------------------------------------------------
        elif mode == 2:
            if hand_result and hand_result.hand_landmarks:
                direction = hand_zone_direction(hand_result.hand_landmarks[0], w, h)
                if direction:
                    snake.set_direction(*direction)
            snake.update()
            cv2.imshow("Snake Game", snake.draw())
            cv2.putText(frame, "Move your hand off-center like a joystick. 'g' = restart",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # -----------------------------------------------------------
        # MODE 3: Pong
        # -----------------------------------------------------------
        elif mode == 3:
            left_y = right_y = None
            if hand_result and hand_result.hand_landmarks:
                for hl, handed in zip(hand_result.hand_landmarks, hand_result.handedness):
                    raw_label = handed[0].category_name
                    display_label = "Right" if raw_label == "Left" else "Left"
                    wrist_y = hl[0].y
                    if display_label == "Left":
                        left_y = wrist_y
                    else:
                        right_y = wrist_y
            pong.set_paddles(left_y, right_y)
            pong.update()
            cv2.imshow("Pong Game", pong.draw())
            cv2.putText(frame, "Left hand = left paddle, Right hand = right paddle",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # -----------------------------------------------------------
        # Recording / logging
        # -----------------------------------------------------------
        if recording and video_writer is not None:
            video_writer.write(frame)
        if logging_on and logger is not None:
            logger.log(hand_result, pose_result)

        # -----------------------------------------------------------
        # HUD
        # -----------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {mode_names[mode]} (0-3)", (w - 380, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if recording:
            cv2.circle(frame, (w - 20, 55), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if logging_on:
            cv2.putText(frame, "LOG", (w - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        status = (f"Hands:{'ON' if detect_hands else 'OFF'}(h) "
                  f"Body:{'ON' if detect_body else 'OFF'}(b) "
                  f"Face:{'ON' if detect_face else 'OFF'}(f) | q=quit")
        cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Advanced Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:  # no key pressed
            continue

        if key == ord('q'):
            break
        elif key == ord('h'):
            detect_hands = not detect_hands
        elif key == ord('b'):
            detect_body = not detect_body
        elif key == ord('f'):
            detect_face = not detect_face
        elif key == ord('g'):
            snake.reset()
            pong.reset()
        elif key == ord('r'):
            recording = not recording
            if recording:
                fname = os.path.join(RECORDINGS_DIR, f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                print(f"Recording started: {fname}")
            else:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Recording stopped.")
        elif key == ord('l'):
            logging_on = not logging_on
            if logging_on:
                fname = os.path.join(RECORDINGS_DIR, f"landmarks_{datetime.now():%Y%m%d_%H%M%S}.csv")
                logger = LandmarkLogger(fname)
                print(f"Landmark logging started: {fname}")
            else:
                if logger is not None:
                    logger.close()
                    logger = None
                print("Landmark logging stopped.")
        elif key in (ord('0'), ord('1'), ord('2'), ord('3')):
            mode = int(chr(key))
            if mode == 1:
                detect_hands = True
            elif mode in (2, 3):
                detect_hands = True
        elif mode == 1 and key == ord('t'):
            asl_training = not asl_training
        elif mode == 1 and asl_training and chr(key).isalpha():
            if hand_result and hand_result.hand_landmarks:
                vec = extract_hand_feature(hand_result.hand_landmarks[0])
                letter = chr(key).upper()
                asl.add_sample(letter, vec)
                print(f"Saved ASL sample for '{letter}' ({len(asl.labels)} total samples)")

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()
    pose_landmarker.close()
    face_landmarker.close()
    if video_writer is not None:
        video_writer.release()
    if logger is not None:
        logger.close()


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------
# SETUP INSTRUCTIONS
# -----------------------------------------------------------------------
# pip install opencv-python mediapipe numpy
#
# Run:
#   python advanced_tracker.py
#
# ASL recognition (mode 1):
#   - Press 't' to enter TRAINING mode.
#   - Hold a hand shape steady and press the matching letter key (a-z)
#     to save it as a training sample. Do this several times per letter
#     from slightly different angles for better accuracy.
#   - Press 't' again to switch to RECOGNIZING mode and see live guesses.
#   - Your samples are saved to asl_training_data.csv next to this
#     script and are reloaded automatically next time you run it, so
#     your "model" keeps improving the more you use it.
#   - Note: J and Z involve motion in real ASL; this classifier only
#     recognizes static hand shapes, so treat those two as a best-effort.
#
# Snake (mode 2): move your hand away from the center of the frame in
#   any direction to steer, like a joystick. Press 'g' to restart after
#   a game over.
#
# Pong (mode 3): show your LEFT hand to control the left paddle and your
#   RIGHT hand to control the right paddle (move them up/down). Works
#   with a friend, both hands in frame at once, or by switching hands
#   yourself.
#
# Recording: press 'r' to start/stop saving an .mp4 of the annotated
#   video to the "recordings" folder. Press 'l' to start/stop logging
#   every frame's raw landmark coordinates to a timestamped CSV file in
#   the same folder, useful for further analysis in Excel/Python.
# -----------------------------------------------------------------------