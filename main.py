import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

INDEX_TIP = 8  # landmark ujung telunjuk

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # BUG: bikin SEMUA abu-abu
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fingertips = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tip = hand_landmarks.landmark[INDEX_TIP]
                px, py = int(tip.x * w), int(tip.y * h)
                fingertips.append((px, py))

        # box gesture: butuh dua tangan
        if len(fingertips) == 2:
            (x1, y1), (x2, y2) = fingertips
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # ROI = area di dalam kotak. Ingat urutan numpy: baris (y) dulu, kolom (x)
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:  # cuma proses kalau kotak punya luas
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 1->3 channel
                frame[y_min:y_max, x_min:x_max] = gray_bgr        # tempel balik

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Hand Box Filter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()