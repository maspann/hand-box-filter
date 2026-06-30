import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

INDEX_TIP = 8  # nomor landmark ujung jari telunjuk di MediaPipe

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape  # ukuran frame, buat konversi koordinat
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingertips = []  # koordinat ujung telunjuk tiap tangan
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                tip = hand_landmarks.landmark[INDEX_TIP]
                # landmark MediaPipe itu 0-1, dikali ukuran frame jadi pixel
                px, py = int(tip.x * w), int(tip.y * h)
                fingertips.append((px, py))
                cv2.circle(frame, (px, py), 10, (0, 255, 0), -1)  # tandai
        
        # kalau dua tangan kedeteksi, gambar kotak antar dua ujung telunjuk
        if len(fingertips) == 2:
            (x1, y1), (x2, y2) = fingertips
            top_left = (min(x1, x2), min(y1, y2))
            bottom_right = (max(x1, x2), max(y1, y2))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
        cv2.imshow("Hand Box Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()