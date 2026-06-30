import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

WRIST = 0
THUMB_TIP, INDEX_TIP = 4, 8

# tiap jari: (ujung, sendi tengah) -> buat cek "jelas terbuka"
FINGER_JOINTS = {
    "thumb":  (4, 2),
    "index":  (8, 6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}
EXTEND_MARGIN = 1.15  # ujung harus 15% lebih jauh dari pergelangan dibanding sendi


# ---------- FILTER: terima ROI (BGR), balikin ROI ukuran sama ----------
def filter_invert(roi):
    return cv2.bitwise_not(roi)

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
    out[edges > 0] = (255, 255, 255)
    return out

def filter_blur(roi):
    return cv2.GaussianBlur(roi, (21, 21), 0)

# REGISTRY: jari -> (label, filter). Urutan list = urutan numpuknya.
FILTER_PIPELINE = [
    ("index",  "PIXELATE", filter_pixelate),
    ("middle", "THERMAL",  filter_thermal),
    ("ring",   "EDGES",    filter_edges),
    ("pinky",  "BLUR",     filter_blur),
    ("thumb",  "INVERT",   filter_invert),
]


# ---------- helper ----------
def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def finger_states(hand_landmarks):
    """Jari mana yang JELAS terbuka. Ngepal -> semua False -> nggak kedeteksi."""
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]
    return {
        name: dist(lm[tip], wrist) > dist(lm[pip], wrist) * EXTEND_MARGIN
        for name, (tip, pip) in FINGER_JOINTS.items()
    }

def order_quad(points):
    """Urutkan 4 titik mengelilingi pusatnya -> poligon nggak nyilang (free-roam)."""
    pts = np.array(points, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)].astype(np.int32)


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

        corners = []      # sudut = ujung jempol+telunjuk, CUMA yang jelas terbuka
        active = set()    # semua jari terbuka dari kedua tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                states = finger_states(hand_landmarks)
                active |= {name for name, up in states.items() if up}
                lm = hand_landmarks.landmark
                if states["thumb"]:
                    corners.append((int(lm[THUMB_TIP].x * w), int(lm[THUMB_TIP].y * h)))
                if states["index"]:
                    corners.append((int(lm[INDEX_TIP].x * w), int(lm[INDEX_TIP].y * h)))

        # GATING: cuma jalan kalau dapet tepat 4 sudut jelas (2 tangan, jempol+telunjuk kebuka)
        if len(corners) == 4:
            quad = order_quad(corners)
            x, y, bw, bh = cv2.boundingRect(quad)
            roi = frame[y:y + bh, x:x + bw]   # view ke frame asli
            if roi.size > 0:
                out = roi.copy()
                labels = []
                for finger, label, fn in FILTER_PIPELINE:
                    if finger in active:          # condition jari terpenuhi?
                        out = fn(out)             # numpuk: output jadi input filter berikutnya
                        labels.append(label)

                # mask poligon (free-roam) dalam koordinat ROI
                mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.fillPoly(mask, [(quad - [x, y]).astype(np.int32)], 255)
                roi[mask > 0] = out[mask > 0]     # tempel filter CUMA di dalam poligon

                cv2.polylines(frame, [quad], True, (0, 255, 0), 2)
                hud = " + ".join(labels) if labels else "no filter"
                cv2.putText(frame, hud, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Hand Box Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()