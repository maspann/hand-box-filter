import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

WRIST = 0
TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
EXTEND_MARGIN = 1.15  # ujung harus cukup jauh dari pergelangan dibanding sendi


# ---------- FILTER: terima ROI (BGR), balikin ROI ukuran sama ----------
def filter_pixelate(roi, blocks=14):
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
    out[edges > 0] = (255, 255, 255)
    return out

def filter_blur(roi):
    return cv2.GaussianBlur(roi, (21, 21), 0)

def filter_invert(roi):
    return cv2.bitwise_not(roi)

# tiap jari -> (label, warna garis, filter). Tiap jari = kotak & filter sendiri.
FINGER_FILTERS = {
    "thumb":  ("INVERT",   (200, 200, 200), filter_invert),
    "index":  ("PIXELATE", (0, 255, 0),     filter_pixelate),
    "middle": ("THERMAL",  (0, 165, 255),   filter_thermal),
    "ring":   ("EDGES",    (255, 0, 255),   filter_edges),
    "pinky":  ("BLUR",     (255, 255, 0),   filter_blur),
}


# ---------- helper ----------
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def extended_fingers(hand_landmarks):
    """Nama jari yang JELAS terbuka. Ngepal/jari nekuk -> nggak masuk -> nggak kedeteksi."""
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]
    out = set()
    for name in TIP_IDS:
        if dist(lm[TIP_IDS[name]], wrist) > dist(lm[PIP_IDS[name]], wrist) * EXTEND_MARGIN:
            out.add(name)
    return out

def square_from_diagonal(p1, p2):
    """2 ujung jari = diagonal -> kotak MIRING (free-roam). Balikin 4 sudut."""
    p1, p2 = np.array(p1, np.float32), np.array(p2, np.float32)
    center = (p1 + p2) / 2
    half = (p2 - p1) / 2
    perp = np.array([-half[1], half[0]])      # tegak lurus, panjang sama -> bikin miring
    return np.array([center + half, center + perp,
                     center - half, center - perp], dtype=np.int32)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # kumpulin ujung jari yang terbuka, dikelompokin PER NAMA JARI (lintas tangan)
        tips = {name: [] for name in FINGER_FILTERS}
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                for name in extended_fingers(hand_landmarks):
                    t = lm[TIP_IDS[name]]
                    tips[name].append((int(t.x * w), int(t.y * h)))

        clean = frame.copy()  # sumber bersih: tiap kotak baca kamera asli, bukan hasil kotak lain

        # tiap jari yang terbuka di KEDUA tangan -> kotaknya sendiri + filternya sendiri
        for name, (label, color, fn) in FINGER_FILTERS.items():
            if len(tips[name]) != 2:      # butuh jari sama di kiri & kanan
                continue
            quad = square_from_diagonal(tips[name][0], tips[name][1])
            x, y, bw, bh = cv2.boundingRect(quad)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = clean[y1:y2, x1:x2]
            filtered = fn(roi)
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.fillPoly(mask, [quad - [x1, y1]], 255)        # poligon miring (free-roam)
            frame[y1:y2, x1:x2][mask > 0] = filtered[mask > 0]  # tempel cuma di dalam poligon

            cv2.polylines(frame, [quad], True, color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Hand Box Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()