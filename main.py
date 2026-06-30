import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

THUMB_TIP, INDEX_TIP = 4, 8
# jari toggle: (landmark ujung, landmark sendi PIP) -> buat deteksi "naik"
TOGGLE_FINGERS = {
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}

# --- FILTER: tiap fungsi terima ROI (BGR), balikin ROI ukuran sama ---
def filter_pixelate(roi, blocks=16):
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // blocks), max(1, h // blocks)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def filter_thermal(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def filter_edges(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    out = roi.copy()
    out[edges > 0] = (255, 255, 255)  # garis putih di atas gambar
    return out

# REGISTRY: jari -> (label, filter). Urutan list = urutan numpuknya.
FILTER_PIPELINE = [
    ("middle", "PIXELATE", filter_pixelate),
    ("ring",   "THERMAL",  filter_thermal),
    ("pinky",  "EDGES",    filter_edges),
]

def fingers_up(hand_landmarks):
    """Set nama jari toggle yang lagi naik (ujung lebih tinggi dari sendi)."""
    lm = hand_landmarks.landmark
    return {name for name, (tip, pip) in TOGGLE_FINGERS.items()
            if lm[tip].y < lm[pip].y}  # y lebih kecil = lebih ke atas

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        corners = []      # 4 sudut kotak
        active = set()    # gabungan jari naik dari semua tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                for tip_id in (THUMB_TIP, INDEX_TIP):
                    corners.append((int(lm[tip_id].x * w), int(lm[tip_id].y * h)))
                active |= fingers_up(hand_landmarks)

        if len(corners) == 4:
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                labels = []
                for finger, label, fn in FILTER_PIPELINE:
                    if finger in active:       # condition terpenuhi?
                        roi = fn(roi)          # numpuk: output jadi input filter berikutnya
                        labels.append(label)
                frame[y_min:y_max, x_min:x_max] = roi
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                hud = " + ".join(labels) if labels else "no filter"
                cv2.putText(frame, hud, (x_min, max(20, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Hand Box Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()