import os
import random
import hashlib
import warnings
from collections import deque
from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
from flask import Flask, Response, redirect, render_template, request, session, url_for
import sqlite3

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from googleapiclient.discovery import build

warnings.filterwarnings("ignore")

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "vibees_secret_key_2024"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "music_emotion", "model.h5")
DB_PATH = os.path.join(BASE_DIR, "vibees.db")

YOUTUBE_API_KEY       = os.getenv("YOUTUBE_API_KEY", "AIzaSyBBC9Syq_gH0KtgK8dY0VPWTG_0-cxJ-aQ")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ── Emotion → genre mapping ───────────────────────────────────────────────────
EMOTION_MUSIC = {
    "Angry":     ["energetic", "metal", "hard rock"],
    "Disgusted": ["motivational", "uplifting"],
    "Fearful":   ["relaxing", "calming", "ambient"],
    "Happy":     ["upbeat", "pop", "dance"],
    "Neutral":   ["lo-fi", "chill", "indie"],
    "Sad":       ["calm", "instrumental", "soul"],
    "Surprised": ["party", "electronic", "fast-paced"],
}

EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Shared state for the live emotion
current_emotion = {"value": "Neutral"}

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email         TEXT,
                phone         TEXT,
                gender        TEXT,
                age           INTEGER,
                dob           TEXT
            )
        """)
        conn.commit()

init_db()

# ── Helpers ───────────────────────────────────────────────────────────────────
def hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def build_model():
    m = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2), Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2), Dropout(0.25),
        Flatten(),
        Dense(1024, activation="relu"), Dropout(0.5),
        Dense(7, activation="softmax"),
    ])
    m.load_weights(MODEL_PATH)
    return m


def fetch_youtube(emotion):
    query = random.choice(EMOTION_MUSIC.get(emotion, ["chill"]))
    resp = youtube.search().list(q=query, part="snippet", maxResults=5).execute()
    return [{"name": i["snippet"]["title"],
             "url": f"https://www.youtube.com/watch?v={i['id']['videoId']}"}
            for i in resp["items"] if i["id"]["kind"] == "youtube#video"]

# ── Video generator ───────────────────────────────────────────────────────────
def gen_frames():
    model = build_model()
    cv2.ocl.setUseOpenCL(False)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=10)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            inp = np.expand_dims(np.expand_dims(roi, -1), 0)
            idx = int(np.argmax(model.predict(inp)))
            buffer.append(EMOTION_DICT[idx])
            current_emotion["value"] = max(set(buffer), key=buffer.count)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (99, 102, 241), 2)
            cv2.putText(frame, current_emotion["value"],
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (99, 102, 241), 2)

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n\r\n")

    cap.release()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        uname = request.form["username"]
        pw    = hash_pw(request.form["password"])
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password_hash=?", (uname, pw)
            ).fetchone()
        if user:
            session["user"] = dict(user)
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username,password_hash,email,phone,gender,age,dob) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (request.form["username"], hash_pw(request.form["password"]),
                     request.form.get("email"), request.form.get("phone"),
                     request.form.get("gender"), request.form.get("age"),
                     request.form.get("dob"))
                )
                conn.commit()
            return redirect(url_for("login"))
        except Exception:
            error = "Username already taken or invalid data."
    return render_template("register.html", error=error)


@app.route("/index")
def index():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/emotion")
def emotion():
    if "user" not in session:
        return redirect(url_for("login"))
    detected = current_emotion["value"]
    youtube_songs = fetch_youtube(detected)
    return render_template("emotion.html",
                           emotion=detected,
                           youtube_songs=youtube_songs)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)
